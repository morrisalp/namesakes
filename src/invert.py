from argparse import Namespace
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Inverter:
    def __init__(self, e4e_model_path, e4e_device, pSp_class, tensor2im_func):
        ckpt = torch.load(e4e_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = e4e_model_path
        opts = Namespace(**opts)

        self.device = e4e_device
        self.tensor2im = tensor2im_func

        self.net = pSp_class(opts)
        self.net.eval()
        self.net.to(self.device)
        # latent_avg is set as a plain attribute (not registered buffer), so move explicitly
        if hasattr(self.net, 'latent_avg') and self.net.latent_avg is not None:
            self.net.latent_avg = self.net.latent_avg.to(self.device)

        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def invert_imgs(self, imgs, batch_size):

        dl = DataLoader(imgs,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: torch.stack([
                self.transforms(img) for img in x
            ]).float().to(self.device)
        )
        L = []
        inv_imgs = []
        for T in dl: # T: shape (b, 3, 256, 256)
            _, batch_latents = self.net(T, randomize_noise=False, return_latents=True)
            del _  # free decoder output from pSp forward before second decode
            # batch_latents: (b, 18, 512)
            L.append(batch_latents.detach())
            inv, _ = self.net.decoder([batch_latents], input_is_latent=True)
            inv_imgs += [
                self.tensor2im(t) for t in inv
            ]
            del inv, _
        L = torch.vstack(L) # shape: (n_imgs, 18, 512)

        Lavg = L.mean(dim=0) # (18, 512)
        inv_avg, _ = self.net.decoder(
            [Lavg.unsqueeze(0)], input_is_latent=True)
        inv_avg_img = self.tensor2im(inv_avg[0])

        return inv_imgs, inv_avg_img, L
