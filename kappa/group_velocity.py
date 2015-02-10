__author__ = 'xinjiang'
import h5py
import sys
import numpy as np

def get_frame(lattice):
    sc=np.array([[ 1.,  0.,  0.],[ 0.,  1.,  0.],[ 0.,  0.,  1.]])
    bcc=np.array([[-0.5,  0.5,  0.5],[0.5, -0.5,  0.5],[ 0.5,  0.5, -0.5]])
    bcc2=np.array([[0.5,  0.5,  0.5],[0.5, -0.5,  -0.5],[ -0.5,  0.5, -0.5]])
    fcc=np.array([[ 0. ,  0.5,  0.5],[ 0.5,  0. ,  0.5],[ 0.5,  0.5,  0. ]])
    sc_dot=np.dot(np.linalg.inv(sc), lattice)
    bcc_dot=np.dot(np.linalg.inv(bcc), lattice)
    bcc2_dot = np.dot(np.linalg.inv(bcc2), lattice)
    fcc_dot=np.dot(np.linalg.inv(fcc), lattice)
    if np.allclose(sc_dot, sc *sc_dot[0,0], atol=1e-5):
        frame=sc
        a=sc_dot[0,0]
    elif np.allclose(bcc_dot, sc *bcc_dot[0,0], atol=1e-5):
        frame=bcc
        a=bcc_dot[0,0]
    elif np.allclose(fcc_dot , sc * fcc_dot[0,0], atol=1e-5):
        frame=fcc
        a=fcc_dot[0,0]
    elif np.allclose(bcc2_dot, sc * bcc2_dot[0,0], atol=1e-5):
        frame = bcc2
        a=bcc2_dot[0,0]
    else:
        a=1.0
        frame=lattice
    return frame, a

class Group_velocity():
    def __init__(self, filename, frame=np.eye(3)):
        self._frame=frame
        self.get_gv_from(filename)

    def get_gv_from(self, filename):
        g=h5py.File(filename)
        rec_lat = np.linalg.inv(self._frame)
        qs=g['q-position'].value
        self._gv=g['group_velocity'].value
        self._weight=g['weight'].value
        self._qpoints=np.dot(rec_lat, qs.T).T
        self._frequency=g['frequency'][:]
        if len(qs) != self._weight.sum():
            print "Error! The group velocity read from file %s should hold no symmetry. Exiting.." %filename
            sys.exit(1)
        g.close()
