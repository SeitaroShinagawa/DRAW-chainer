import numpy
from chainer import cuda,Variable

class XP:
    __lib = None

    @staticmethod
    def set_library(args):
        if args.gpu>=0:
            XP.__lib = cuda.cupy
            cuda.get_device(args.gpu).use()
        else:
            XP.__lib = numpy

    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype))

    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)
    
    @staticmethod
    def __arange(shape, dtype):
        return Variable(XP.__lib.arange(shape, dtype=dtype))

    @staticmethod
    def farange(shape):
        return XP.__arange(shape, XP.__lib.float32)

    @staticmethod
    def __nonzeros(shape, dtype, val):
        return Variable(val * XP.__lib.ones(shape, dtype=dtype))

    @staticmethod
    def fnonzeros(shape, val=1):
        return XP.__nonzeros(shape, XP.__lib.float32, val)

    @staticmethod
    def __array(array, dtype):
        return Variable(XP.__lib.array(array, dtype=dtype))

    @staticmethod
    def iarray(array):
        return XP.__array(array, XP.__lib.int32)

    @staticmethod
    def farray(array):
        return XP.__array(array, XP.__lib.float32)
