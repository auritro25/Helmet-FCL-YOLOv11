import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.cfg import get_cfg



class FedProxTrainer(DetectionTrainer):


    def __init__(
        self,
        cfg=None,
        overrides=None,
        _callbacks=None,
        **kwargs
    ):


        self.global_weights = None
        self.mu = 0.01


        if cfg is None:
            cfg = get_cfg()


        if overrides is None:
            overrides = {}


        super().__init__(
            cfg=cfg,
            overrides=overrides,
            _callbacks=_callbacks
        )



    def optimizer_step(self):


        if self.global_weights is not None:


            prox_loss = torch.zeros(

                1,

                device=self.device

            )



            for name, param in self.model.named_parameters():


                if name in self.global_weights:


                    global_param = (

                        self.global_weights[name]

                        .to(self.device)

                    )


                    prox_loss += torch.sum(

                        (

                            param

                            -

                            global_param

                        )

                        ** 2

                    )



            fedprox_term = (

                self.mu / 2

            ) * prox_loss



            print(

                "FedProx penalty:",

                fedprox_term.item()

            )



            fedprox_term.backward()



        super().optimizer_step()