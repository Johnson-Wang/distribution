__author__ = 'xinjiang'
import numpy as np
import os
import sys
from string import maketrans


def print_error():
    print """  ___ _ __ _ __ ___  _ __
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |
 \___|_|  |_|  \___/|_|
"""

def file_exists(filename, log_level):
    if os.path.exists(filename):
        return True
    else:
        error_text = "%s not found." % filename
        print_error_message(error_text)
        if log_level > 0:
            print_error()
        sys.exit(1)

def print_error_message(message):
    print
    print message
    print_error()
    sys.exit(1)

def print_warning_message(message):
    print
    print "WARNING:",message


class Settings():
    def __init__(self,
                 filename=None,
                 options=None,
                 option_list=None):
        self._filename=filename
        if self._filename is None:
            self._kappa_mode="c" #calculating thermal conductivity
        else:
            self._kappa_mode="r" #reading thermal conductivity
        self._options=options
        self._optlist=option_list

        self._temperatures=None#np.array([300], dtype=float)
        self._quiet=False
        self._verbose=False
        self._is_num = False #is non-uniform mesh?
        self._is_fix_sigma = False
        self._poscar="POSCAR"
        self._cutoff_lifetime=1e-4
        self._mesh = None
        self._gv_mesh_shift=np.array([0.5, 0.5, 0.5], dtype=float)
        self._gv_shift_file=None
        self._iso_file = None
        self._thick=None
        self._rough=None
        self._specularity=None
        self._language="C"
        self._dist=None
        self._dest=None
        self._atool=None
        self._aobject=None
        self._bi=slice(None) #band index
        self._sigma=1.0
        self._seg=200
        self._is_plot=False
        self._is_cumulative=False
        self._is_cahill_pohl=False
        self._is_save=False
        self._is_log=[False,False]
        self._is_scatter=False
        self._dirt=(0,0) # thermal conductivity direction tensor
        self._dirv=None # group velocity direciton vector
        self._dirw = -1 # only for boundary scattering. -1 respresents all directions
        self._vfactor=1.0
        self._nosym = False
        self._thick_unit = None
        self._rough_unit = None

        if options is not None and option_list is not None:
            if options.configure is not None:
                conf=self.read_options_from_file(options.configure)
            else:
                conf={}
            self.read_options(conf)

    def read_options_from_file(self, filename):
        file_exists(filename, 1)
        file = open(filename, 'r')
        confs = {}
        is_continue = False
        for line in file:
            if line.strip() == '':
                is_continue = False
                continue

            if line.strip()[0] == '#':
                is_continue = False
                continue

            if is_continue:
                confs[left] += line.strip()
                confs[left] = confs[left].replace('+++', ' ')
                is_continue = False

            if line.find('=') != -1:
                left, right = [x.strip().lower() for x in line.split('=')]
                if left == 'band_labels':
                    right = [x.strip() for x in line.split('=')][1]
                confs[left] = right

            if line.find('+++') != -1:
                is_continue = True
        return confs

    def read_options(self, conf):
        for opt in self._optlist:
            if opt.dest == 'temperatures':
                if self._options.temperatures is not None:
                    t_string=self._options.temperatures
                    if t_string.find(":") ==-1:
                        self._temperatures=np.array(t_string.replace(",", " ").split(), dtype=float)
                    else:
                        self._temperatures=eval("np.arange(10000)[%s]"%t_string).astype(float)
            if opt.dest == "verbose":
                self._verbose=self._options.verbose
            if opt.dest=="quiet":
                self._quiet=self._options.quiet
            if opt.dest=="poscar":
                self._poscar=self._options.poscar
            if opt.dest=="cutoff_lifetime":
                self._cutoff_lifetime=self._options.cutoff_lifetime
            if opt.dest=="gv_mesh_shift":
                shift=self._options.gv_mesh_shift.replace(",", " ").split()
                self._gv_mesh_shift=np.array(shift, dtype=float)
            if opt.dest == "iso_file":
                filename=self._options.iso_file
                if filename is not None:
                    if file_exists(filename, 1):
                        self._iso_file = filename

            if opt.dest=="gv_shift_file":
                filename=self._options.gv_shift_file
                if filename is not None:
                    if file_exists(filename,1):
                        self._gv_shift_file=filename
            if opt.dest=="thick":
                thick=self._options.thick
                if thick is not None:
                    if thick.lower().find("nm") != -1:
                        self._thick_unit = 10
                        thick = thick.lower().replace("nm", "")
                    elif thick.lower().find("a") != -1:
                        self._thick_unit = 1 # standard is Angstrom
                        thick = thick.lower().replace("a", "")
                    self._thick=np.array(thick.replace(",", " ").split(), dtype=float)

            if opt.dest=="mesh":
                mesh = self._options.mesh
                if mesh is not None:
                    self._mesh = np.fromstring(mesh.replace(",", " "), sep=" ", dtype=np.int)

            if opt.dest=="rough":
                rough=self._options.rough
                if rough is not None:
                    if rough.lower().find("nm") != -1:
                        self._rough_unit = 10
                        rough = rough.lower().replace("nm", "")
                    elif rough.lower().find("a") != -1:
                        self._rough_unit = 1
                        rough = rough.lower().replace("a", "")
                    self._rough=np.array(rough.replace(",", " ").split(),dtype=float)

            if opt.dest=="specularity":
                p=self._options.specularity
                if p is not None:
                    self._specularity=np.array(p.replace(",", " ").split(), dtype="double")

            if opt.dest=="language":
                l=self._options.language
                if l.strip().upper()[0] == "P":
                    self._language="P"
                elif l.strip().upper()[0]=="C":
                    self._language="C"
                else:
                    print_error_message("Language can only be set as python or C")

            if opt.dest=="direction_tensor":
                d=self._options.direction_tensor
                tran=maketrans("xyz", "012")
                if d is not None:
                    try:
                        self._dirt=tuple(map(int, d.translate(tran)))
                    except ValueError:
                        print_error_message("The direction can only be set as characters 'x', 'y' and 'z'")
                    if len(self._dirt) == 1:
                        self._dirt = (self._dirt[0],self._dirt[0])
                    elif len(self._dirt) != 2:
                        print_error_message("The direction is a second-order tensor, please recheck the settings!")

            if opt.dest == "direction_vector":
                d=self._options.direction_vector
                tran=maketrans("xyz", "012")
                if d is not None and len(d) >0 :
                    try:
                        self._dirv = int(d.translate(tran)[0])
                    except ValueError:
                        print_error_message("The direction can only be set as characters 'x', 'y' and 'z'")

            if opt.dest == "direction_width":
                d=self._options.direction_width
                tran=maketrans("xyz", "012")
                if d is not None and len(d) >0 :
                    try:
                        self._dirw = int(d.translate(tran)[0])
                    except ValueError:
                        print_error_message("The direction can only be set as characters 'x', 'y' and 'z'")

            if opt.dest == "vfactor":
                if self._options.vfactor is not None:
                    self._vfactor = self._options.vfactor

            if opt.dest == "nosym":
                if self._options.nosym is not None:
                    self._nosym = self._options.nosym

            if opt.dest == "cahill_pohl":
                if self._options.cahill_pohl is not None:
                    self._is_cahill_pohl = self._options.cahill_pohl

            if opt.dest=="distribution":
                d=self._options.distribution
                if d is not None:
                    dist=d.split("|")
                    if len(dist)>1:
                        self._dest=dist[1].strip().upper()[0]
                    else:
                        self._dest="K" # distribution destrination
                    dist=dist[0].replace(",", " ").split()
                    if len(dist)==0:
                        print_error_message("You must choose a property for the distribution")
                    self._dist=[d.strip().upper()[0] for d in dist]

            if opt.dest == "average":
                a = self._options.average
                if a is not None:
                    ave = a.split("|")
                    if len(ave)>1:
                        self._atool = ave[1].strip().upper()[0]
                    else:
                        self._atool = "B" # average over Brillouin zone
                    ave = ave[0].replace(","," ").split()
                    if len(ave)==0:
                        print_error_message("You must choose a property to get its average")
                    self._aobject = [a.strip().upper()[0] for a in ave]

            if opt.dest == "is_num":
                if self._options.is_num is not None:
                    self._is_num = self._options.is_num

            if opt.dest == "is_fix_sigma":
                if self._options.is_fix_sigma is not None:
                    self._is_fix_sigma = self._options.is_fix_sigma

            if opt.dest=="band_index":
                if self._options.band_index is not None:
                    exec "self._bi=np.s_[%s]"%self._options.band_index
                if type(self._bi) == int:
                    self._bi=[self._bi]

            if opt.dest=="sigma":
                if self._options.sigma is not None:
                    self._sigma=self._options.sigma

            if opt.dest=="seg":
                if self._options.seg is not None:
                    self._seg = self._options.seg

            if opt.dest=="is_plot":
                self._is_plot=self._options.is_plot

            if opt.dest=="is_cumulative":
                self._is_cumulative = self._options.is_cumulative

            if opt.dest=="is_save":
                self._is_save=self._options.is_save

            if opt.dest=="is_log":
                is_log = self._options.is_log
                if is_log is not None:
                    if is_log.upper().find("X")!=-1:
                        self._is_log[0] = True
                    if is_log.upper().find("Y")!=-1:
                        self._is_log[1] = True

            if opt.dest=="is_scatter":
                self._is_scatter=self._options.is_scatter