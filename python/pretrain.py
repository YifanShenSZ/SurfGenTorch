import logging
from pathlib import Path
from typing import List

import torch
import torch.nn.functional
import torch.optim

import utility
import SSAIC
import AbInitio
import DimRed

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig()
logger.setLevel("DEBUG")

@torch.no_grad()
def RMSD(irred:int, net:DimRed.DimRedNet, data_set:List) -> torch.tensor:
    e = 0
    for data in data_set:
        e += torch.nn.functional.mse_loss(
            data.SAIgeom[irred], net.forward(data.SAIgeom[irred]),
            reduction='sum')
    e /= len(data_set)
    e /= sum(SSAIC.NSAIC_per_irred)
    e = torch.sqrt(e)
    return e

def pretrain(irred:int, max_depth:int, data_set_file:List[Path], chk:Path=None, opt="SGD", epoch=1000) -> None:
    # contains
    def train() -> None:
        # contains
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = 0
            for data in batch:
                loss += torch.nn.functional.mse_loss(
                    data.SAIgeom[irred], net.forward(data.SAIgeom[irred]),
                    reduction='sum')
            loss.backward()
            return loss
        for iepoch in range(epoch):
            for batch in geom_loader:
                optimizer.step(closure=closure)
            if iepoch % follow == follow - 1:
                print("epoch = %d, RMSD = %e" % (iepoch+1, RMSD(irred, net, GeomSet.example)))
                torch.save({"net":net.state_dict(), "opt":optimizer.state_dict()},
                    "pretrain-"+str(irred)+"_epoch-"+str(iepoch+1)+".pt")
    def generate_optimizer():
        if opt == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
        elif opt == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                lr=0.01)
        elif opt == "LBFGS":
            optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, net.parameters()),
                line_search_fn="strong_wolfe")
        return optimizer

    print("Start pretraining")

    net = DimRed.DimRedNet(irred, max_depth)
    net.to(torch.double)

    data_set = utility.verify_data_set(data_set_file)
    GeomSet = AbInitio.read_GeomSet(data_set)
    print("Number of geometries = %d" % GeomSet.__len__())
    if opt == "LBFGS":
        geom_loader = torch.utils.data.DataLoader(GeomSet,
        batch_size=GeomSet.__len__(),
        collate_fn=AbInitio.collate)
    else:
        geom_loader = torch.utils.data.DataLoader(GeomSet,
        shuffle=True,
        collate_fn=AbInitio.collate)
    logger.info("batch size = %d", geom_loader.batch_size)

    follow = int(epoch / 10)
    if chk != None:
        print("Continue from checkpoint")
        checkpoint = torch.load(chk)
        net.load_state_dict(checkpoint["net"], strict=False)
        net.train()
        if len(net.state_dict()) > len(checkpoint["net"]):
            print("Warm start a deeper net: Freeze inherited layers, then train")
            for key in checkpoint["net"]:
                strs = key.split('.')
                if strs[0] == 'fc':
                    net.fc[int(strs[1])].weight.requires_grad = False
                else:
                    net.fc_inv[int(strs[1])].weight.requires_grad = False
            print("Number of trainable parameters = %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))
            optimizer = generate_optimizer()
            train()
            print("Defreeze inherited layers, then train again")
            for p in net.parameters(): p.requires_grad = True
            print("Number of trainable parameters = %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))
            optimizer = generate_optimizer()
            train()
        elif len(net.state_dict()) == len(checkpoint["net"]):
            print("Continue with the same net")
            optimizer = generate_optimizer()
            if (opt == "Adam"  and "betas"          in checkpoint["opt"]['param_groups'][0]) \
            or (opt == "SGD"   and "nesterov"       in checkpoint["opt"]['param_groups'][0]) \
            or (opt == "LBFGS" and "tolerance_grad" in checkpoint["opt"]['param_groups'][0]):
                optimizer.load_state_dict(checkpoint["opt"])
            train()
        else:
            print("Warm start a shallower net")
            optimizer = generate_optimizer()
            train()
    else:
        print("Training...")
        optimizer = generate_optimizer()
        train()