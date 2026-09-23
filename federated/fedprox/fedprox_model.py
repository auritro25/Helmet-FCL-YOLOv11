import torch


class FedProxLossWrapper:


    def __init__(
        self,
        model,
        global_weights,
        mu=0.01
    ):

        self.model = model

        self.original_loss = model.loss

        self.global_weights = global_weights

        self.mu = mu



    def __call__(
        self,
        batch,
        preds=None
    ):


        # Original YOLO loss

        loss_output = self.original_loss(
            batch,
            preds
        )


        if isinstance(loss_output, tuple):

            loss = loss_output[0]
            items = loss_output[1]

        else:

            loss = loss_output
            items = None



        prox = torch.zeros(
            1,
            device=loss.device
        )



        for name, param in self.model.named_parameters():


            if name in self.global_weights:


                global_param = (

                    self.global_weights[name]
                    .to(param.device)

                )


                prox += torch.sum(

                    (param - global_param) ** 2

                )



        loss = (

            loss

            +

            (self.mu / 2)

            *

            prox

        )



        if items is not None:

            return loss, items


        return loss