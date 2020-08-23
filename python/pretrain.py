from pathlib import Path
from typing import List
import torch
import torch.nn.functional
import torch.optim

import SSAIC
import AbInitio
import DimRed

@torch.no_grad()
def RMSD(irred:int, net:DimRed.DimRedNet, GeomSet:List) -> torch.tensor:
    e = 0
    for geom in GeomSet:
        e += torch.nn.functional.mse_loss(
            net.forward(geom.SAIgeom[irred]), geom.SAIgeom[irred],
            reduction='sum')
    e /= len(GeomSet)
    e /= SSAIC.NSAIC_per_irred[irred]
    e = torch.sqrt(e)
    return e

def pretrain(irred:int, max_depth:int, freeze:int,
data_set:List[Path],
chk:List[Path]=None,
opt="Adam", epoch=1000) -> None:
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

    net = DimRed.DimRedNet(SSAIC.NSAIC_per_irred[irred], max_depth)
    net.to(torch.double)
    if chk != None:
        net_chk = torch.load(chk[0])
        net.load_state_dict(net_chk, strict=False)
        net.train()
    net.freeze(freeze)
    print("Number of trainable parameters = %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))

    GeomSet = AbInitio.read_GeomSet(data_set)
    print("Number of geometries = %d" % GeomSet.__len__())
    if opt == "LBFGS":
        geom_loader = torch.utils.data.DataLoader(GeomSet,
        batch_size=GeomSet.__len__(),
        collate_fn=AbInitio.collate)
    else:
        geom_loader = torch.utils.data.DataLoader(GeomSet,
        batch_size=8,
        shuffle=True,
        collate_fn=AbInitio.collate)
    print("batch size = %d" % geom_loader.batch_size)

    follow = int(epoch / 10)
    optimizer = generate_optimizer()
    if chk != None:
        if len(chk) > 1:
            opt_chk = torch.load(chk[1])
            optimizer.load_state_dict(opt_chk)

    for iepoch in range(epoch):
        for batch in geom_loader:
            optimizer.step(closure=closure)
        if iepoch % follow == follow - 1:
            print("epoch = %d, RMSD = %e" % (iepoch+1, RMSD(irred, net, GeomSet.example)))
            torch.save(net.state_dict(), "pretrain_net_"+str(iepoch+1)+".pt")
            torch.save(optimizer.state_dict(), "pretrain_opt_"+str(iepoch+1)+".pt")