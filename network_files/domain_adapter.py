import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DAdapter(nn.Module):
    def __init__(self, num_convs=4, in_channels=256, grad_reverse_lambda=0.1, grl_applied_domain='both', adapter_lambda=0.1):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAdapter, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            # dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.BatchNorm2d(in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    # torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.kaiming_uniform_(l.weight)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain
        self.adapter_lambda = adapter_lambda

    def source_forward(self, feature, target=1, domain='source'):
        assert target == 1
        assert domain == 'source'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss

    def target_forward(self, feature, target=0, domain='target'):
        assert target == 0
        assert domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss

    def forward(self, feature, order='st'):
        assert order == 'st' or order == 'ts'

        batch = feature.shape[0]
        batch_num = int(batch/2)
        if order == 'st':
            loss_s = self.source_forward(feature=feature[:batch_num, ::])
            loss_t = self.target_forward(feature=feature[batch_num:, ::])
        else:
            loss_t = self.target_forward(feature=feature[:batch_num, ::])
            loss_s = self.source_forward(feature=feature[batch_num:, ::])
        loss = self.adapter_lambda * (loss_t+loss_s)
        return dict(loss_da=loss)