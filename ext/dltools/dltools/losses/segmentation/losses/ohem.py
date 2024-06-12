from seglib.calculus import smul
from seglib.calculus.losses import _BaseLoss


class OHEM_Wrapper(_BaseLoss):
    """
    Class to wrap losses and using online hard examples mining.
    Args:
        @param Crit - instance of criterion class
        @param thres - how many loss elements (elements of non reducted loss tensor) will be used to calculate loss
        relatively to minimum achivable reducef.
    Note:
        This wrapper will set the Loss reducef to 'n',or min_reduction. and after ohem will use old reducef.
        If loss has attr 'min_reduction', because 'n' reducef can't be used,then used only 'min_reduction',
         and after sets loss object instance reducef attribute back.
    """

    # TODO: Fix it
    def __init__(self, Crit, thres):

        self._thres = thres
        self._old_red = Crit.rstr
        self.Crit = Crit

        super(OHEM_Wrapper, self).__init__(reduction="n")
        if not hasattr(self.Crit, "min_reduction"):

            self.Crit.rstr = self.reducef

        else:
            super(type(self.Crit), self.Crit).__init__(self.Crit.min_reduction)

    #             self._old_red = self.reducef

    def __call__(self, *args):

        # print(self.reducef, type(self.reducef))
        # print(self.Crit.rstr, type(self.Crit.rstr))
        x = self.Crit(*args)
        # print('x', x, 'shape', x.shape)
        thres_point = int(smul(x.shape) * self._thres)
        _, indicies = x.view(smul(x.shape)).sort()
        # print(thres_point, smul(x.shape))
        # print(indicies)
        mask = indicies[:-thres_point]
        # print(mask)
        x.view(smul(x.shape))[mask] = 0
        x = self._old_red(x)
        return x
