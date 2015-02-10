__author__ = 'xinjiang'
import numpy as np
import matplotlib.pyplot as plt
from phonopy.units import EV, Angstrom
from settings import print_warning_message, print_error_message
np.seterr(divide="ignore")

symbol_converion={"F": "Frequency (THz)",
                  "K": "Kappa (W/m-K)",
                  "M": "MFP (nm)",
                  "W": "Wave length (nm)",
                  "R": "Relaxation time (ps)",
                  "N": "Normal relaxation time (ps)",
                  "U": "Umklapp relaxation time (ps)",
                  "G": "group velocity (Km/s)",
                  "C": "Heat capacity (J/m^3-K)"}


def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def get_direction_index(direc_tuple):
    if direc_tuple==None:
        direc_tuple=(0,0)
    nine_dir = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1), (2,1), (2,0), (1,0)]
    ndir=nine_dir.index(direc_tuple)
    ndir -= (ndir>=6)*3 # considering the equivalence between symmetric pairs
    return ndir

def allocate_space(prop, seg, is_log = False, sigma=0):
    if is_log:

        mi = np.log10(prop[np.nonzero(prop)].min())
        ma = np.log10(prop[np.nonzero(prop)].max())
        if sigma == 0:
            sigma = (ma - mi) / seg
        seg_scheme = np.logspace(mi, ma + 3 * sigma, num=seg, endpoint=True)
    else:
        ma=prop.max()
        mi=prop.min()
        if sigma == 0:
            sigma = (ma - mi) / seg
        seg_scheme = np.linspace(mi, ma + 3 * sigma, num=seg, endpoint=True)
    return seg_scheme

def distribution_py(prop, dest, cast,sigma=0.2, is_log=False):
    for s, p in enumerate(cast[:,0]):
        for i,j in list(np.ndindex(prop.shape)):
            cast[s, 1]+=dest[i,j]* gaussian(prop[i,j]-p, sigma=sigma)

def delta_summation(freq_cast, freq_all, weight, sigma=0.2):
    delta_sum=np.zeros_like(freq_cast)
    for i,f in enumerate(freq_cast):
        delta_sum[i]=np.sum(gaussian(freq_all-f, sigma=sigma)*weight.reshape(-1,1))
    return delta_sum

def write_plot_scatter(property,
                       destination,
                       x1, x2,
                       thick=None,
                       rough=None,
                       p=None,
                       temp=None,
                       is_log=[False,False],
                       is_plot=False,
                       is_save=False,
                       outname=""):
    outname=property+'_'+destination
    if temp is not None:
        outname+="-T%d"%temp
    if thick is not None:
        outname+="-t%.1E"%thick
    if rough is not None:
        outname+="-r%.1f"%rough
    if p is not None:
        outname+="-p%3.2f"%p
    outname+="-scatter"
    f=file(outname+".dat", "w")
    numb=x2.shape[-1] # number of bands
    f.write("%4s\t"%"i")
    for i in range(numb):
        f.write("%15s\t"%("%s-bi%d"%(property,i)))
        f.write("%15s\t"%("%s-bi%d"%(destination,i)))
    f.write("\n")
    f.write("\n")
    for i, (x,y) in enumerate(zip(x1,x2)):
        f.write("%4d\t" %i)
        for j in range(numb):
            f.write("%15.8f\t%15.8f\t" %(x[j],y[j]))
        f.write("\n")
    f.close()
    if is_plot:
        plt.figure()
        plt.scatter(x1, x2)
        plt.title(outname)
        plt.xlabel(symbol_converion[property])
        plt.ylabel(symbol_converion[destination])
        if is_log[0]:
            min=x1[np.nonzero(x2)].min()
            max=x1.max()
            plt.xlim((min,max))
            plt.xscale('log')
        if is_log[1]:
            min=x2[np.nonzero(x2)].min()
            max=x2.max()
            plt.ylim((min,max))
            plt.yscale('log')
        plt.show()
        if is_save:
            plt.savefig(outname+".ps")



def plot_save(cast, name="distribution", is_save=False, is_log=[False,False], xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(cast[:,0], cast[:,1])
    plt.title(name)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if is_log[0]:
        x=cast[:,0]
        min=x[np.nonzero(x)].min()
        max=x.max()
        plt.xlim((min,max))
        plt.xscale('log')
    if is_log[1]:
        y=cast[:,1]
        min = y[np.nonzero(y)].min()
        max=y.max()
        plt.ylim((min,max))
        plt.yscale('log')
    plt.show()
    if is_save:
        plt.savefig(name+".pdf")

def write_distribution(property,
                       destination,
                       cast,
                       temp=None,
                       thick=None,
                       rough=None,
                       p=None,
                       sigma=None,
                       is_plot=False,
                       is_save=False,
                       is_cumulative=False,
                       is_log=[False,False]):
    outname=property+'_'+destination
    if temp is not None:
        outname+="-T%d"%temp
    if thick is not None:
        outname+="-t%.1E"%thick
    if rough is not None:
        outname+="-r%.1f"%rough
    if p is not None:
        outname+="-p%3.2f"%p
    if sigma is not None:
        outname+="-s%.1f"%sigma
    if is_cumulative:
        outname+="-sum"
    if is_plot:
        plot_save(cast,
                  name=outname,
                  is_save=is_save,
                  is_log=is_log,
                  xlabel=symbol_converion[property],
                  ylabel=symbol_converion[destination])
    outname+=".dat"
    f=open(outname, "w")
    for i, (p, d) in enumerate(cast):
        f.write("%15.8f\t%15.8f\n" %(p,d))
    f.close()


class Distribution():
    def __init__(self,
                 tc=None,
                 distribution=None,
                 destination="K",
                 bi=None,
                 sigma=1.0,
                 is_fix_sigma=False,
                 seg=200,
                 language="C",
                 is_plot=False,
                 is_save=False,
                 is_log=[False,False],
                 is_scatter=False,
                 is_cumulative=False,
                 direct=None,
                 direcv=None):
        self._tc = tc # thermal conductivity or group velocity
        self._dist= distribution
        self._dest = destination
        self._screen = None
        self._bi=bi
        self._cast=None
        self._sigma=sigma
        self._seg=seg
        self._real_sigma=None
        self._lang=language
        self._is_plot=is_plot
        self._is_save=is_save
        self._is_log=is_log
        self._is_scatter=is_scatter
        self._is_fix_sigma = is_fix_sigma
        self._is_cumulative=is_cumulative
        self._dirt= get_direction_index(direct) # direction tensor and direction vector
        self._dirv = direcv

    def cast_fun(self):
        for pro in self._dist:
            self._property=pro
            if pro=="F":
                self._screen=self._tc._frequency[:,self._bi]
            elif pro=="W": # wavelength
                if self._dirv is None:
                    qpoints= np.sqrt(np.sum(self._tc._qpoints**2, axis=-1))
                else:
                    qpoints = self._tc._qpoints[:,self._dirv]
                wavelength=np.where(qpoints>1e-5, self._tc._a/(qpoints), 0) / 10 # unit in nm
                self._screen=wavelength.repeat(self._tc._frequency.shape[-1]).reshape(self._tc._frequency.shape)[:,self._bi]
            if self._dest=="K" or \
                            self._dest=="R" or\
                            self._dest == "N" or \
                            self._dest == "U" or \
                            self._dest=="M" or\
                            self._dest=="C":
                if "_thick" in self._tc.__dict__.keys():
                    self.cast_of_kappa_thin()
                else:
                    self.cast_of_kappa_bulk()
            elif self._dest=="G":
                self.cast_of_gv()

    def cast_of_gv(self):
        if self._dirv is not None:
            gv = np.abs(self._tc._gv[..., self._dirv]) / 10
        else:
            gv= np.sqrt(np.sum((self._tc._gv **2), axis=-1))/10
        self._projector = gv[:, self._bi]
        # get the absolute value of group velocity, 10 is the conversion factor from 100m/s to Km/s
        if self._is_scatter:
            write_plot_scatter(self._property,
                               self._dest,
                               self._screen,
                               self._projector,
                               is_log= self._is_log,
                               is_plot=self._is_plot,
                               is_save=self._is_save)
        else:
            self.cast_one()
            self.set_delta_summation()
            write_distribution(self._property,
                               self._dest,
                               self._cast,
                               sigma=self._real_sigma,
                               is_plot=self._is_plot,
                               is_save=self._is_save,
                               is_log=self._is_log)

    def set_delta_summation(self):
        if self._lang=="C":
            try:
                import _kthin as kt
                d_sum=np.zeros_like(self._cast[:,0])
                kt.sumdelta((self._cast[:,0]).copy(), self._screen.copy(), d_sum, self._tc._weight, self._real_sigma)
            except ImportError:
                print_warning_message("the C module _kthin is not imported, python is used instead")
                d_sum=delta_summation(self._cast[:,0],self._screen,self._tc._weight, sigma=self._real_sigma)
        else:
            d_sum=delta_summation(self._cast[:,0], self._screen,self._tc._weight, sigma=self._real_sigma)
        self._cast[:,1] /=d_sum

    def cast_of_kappa_thin(self):
        if self._dirv is not None:
            gv = np.abs(self._tc._gv[:, self._bi, self._dirv]) / 10
        else:
            gv= np.sqrt(np.sum((self._tc._gv[:,self._bi] **2), axis=-1))/10
        for j in np.arange(self._tc._kappa.shape[1]): # roughness or specularity
            for k, temp in enumerate(self._tc._ts):
                for i, thick in enumerate(self._tc._thick):
                    if self._property == "G": #group velocity
                        self._screen = gv
                    if self._property == "M":#mean free path
                        # gv= np.sqrt(np.sum((self._tc._gv **2), axis=-1))[:,self._bi]
                        gamma=self._tc._gamma[i,j,:,k,self._bi]
                        mfp=np.where(gamma>1e-8, gv/ (2 * 2 * np.pi * gamma), 0) # unit in nm
                        self._screen=mfp
                    elif self._property=="R":
                        gamma=self._tc._gamma[:,k,self._bi]
                        self._screen=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
                    elif self._property=="N": # normal gamma
                        gamma=self._tc._gamma_N[:,k,self._bi]
                        self._screen=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0)
                    elif self._property=="U": # Umklapp gamma
                        gamma=self._tc._gamma_U[:,k,self._bi]
                        self._screen=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0)
                    if self._dest =="K":
                        self._projector=self._tc._kappa[i,j,:,k, self._bi]
                    elif self._dest=="R":# relaxation time distribution
                        gamma=self._tc._gamma[i,j,:,k,self._bi]
                        self._projector=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
                    elif self._dest=="N":# relaxation time distribution
                        gamma=self._tc._gamma_N[i,j,:,k,self._bi]
                        self._projector=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
                    elif self._dest=="U":# relaxation time distribution
                        gamma=self._tc._gamma_U[i,j,:,k,self._bi]
                        self._projector=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
                    elif self._dest=="M": #mean free path
                        gamma=self._tc._gamma[i,j,:,k,self._bi]
                        mfp=np.where(gamma>1e-8, gv/ (2 * 2 * np.pi * gamma), 0) # unit in nm
                        self._projector=mfp
                    elif self._dest=="C":#heat capacity
                        V=self._tc._cell.get_volume()
                        self._projector=self._tc._heat_capacity[:,k,self._bi]/V *EV/(Angstrom)**3
                    if self._is_scatter:
                        if self._tc._rough is not None:
                            write_plot_scatter(self._property,
                                               self._dest,
                                               self._screen,
                                               self._projector,
                                               temp=temp,
                                               thick=thick,
                                               rough=self._tc._rough[j],
                                               is_log= self._is_log,
                                               is_plot=self._is_plot,
                                               is_save=self._is_save)
                        else:
                            write_plot_scatter(self._property,
                                               self._dest,
                                               self._screen,
                                               self._projector,
                                               temp=temp,
                                               thick=thick,
                                               p=self._tc._specularity[j],
                                               is_log= self._is_log,
                                               is_plot=self._is_plot,
                                               is_save=self._is_save)
                    else:
                        if self._sigma < 1e-5:
                            self.statistical_distribution()
                        else:
                            self.cast_one()
                            if self._dest!="K":
                                self.set_delta_summation()
                        if self._tc._rough is not None:
                            write_distribution(self._property,
                                               self._dest,
                                               self._cast,
                                               temp=temp,
                                               thick=thick,
                                               rough=self._tc._rough[j],
                                               sigma=self._real_sigma,
                                               is_plot=self._is_plot,
                                               is_save=self._is_save,
                                               is_cumulative=self._is_cumulative,
                                               is_log=self._is_log)
                        else:
                            write_distribution(self._property,
                                               self._dest,
                                               self._cast,
                                               temp=temp,
                                               thick=thick,
                                               p=self._tc._specularity[j],
                                               sigma=self._real_sigma,
                                               is_plot=self._is_plot,
                                               is_save=self._is_save,
                                               is_cumulative=self._is_cumulative,
                                               is_log=self._is_log)


    def cast_of_kappa_bulk(self):
        if self._dirv is not None:
            gv = np.abs(self._tc._gv[:, self._bi, self._dirv]) / 10
        else:
            gv= np.sqrt(np.sum((self._tc._gv[:,self._bi] **2), axis=-1))/10
        for k,temp in enumerate(self._tc._ts):
            self._temp = temp
            if self._property == "G": #group velocity
                self._screen = gv
            if self._property == "M":#mean free path
                # gv= np.sqrt(np.sum((self._tc._gv **2), axis=-1))[:,self._bi]
                gamma=self._tc._gamma[:,k,self._bi]
                mfp=np.where(gamma>1e-8, gv/ (2*2*np.pi*gamma), 0) # unit in nm
                self._screen=mfp
            elif self._property=="R":
                gamma=self._tc._gamma[:,k,self._bi]
                self._screen=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
            elif self._property=="N":
                gamma=self._tc._gamma_N[:,k,self._bi]
                self._screen=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
            elif self._property=="U":
                gamma=self._tc._gamma_U[:,k,self._bi]
                self._screen=np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
            if self._dest=="K":
                self._projector=self._tc._kappa[:,k, self._bi, self._dirt]# considering 6 directions [xx, yy, zz, yz, xz, xy]
            elif self._dest=="R":
                gamma=self._tc._gamma[:,k,self._bi]
                self._projector=np.where(gamma>1e-8, 1/(2*2*np.pi*gamma), 0)
            elif self._dest=="N":
                gamma=self._tc._gamma_N[:,k,self._bi]
                self._projector=np.where(gamma>1e-8, 1/(2*2*np.pi*gamma), 0)
            elif self._dest=="U":
                gamma=self._tc._gamma_U[:,k,self._bi]
                self._projector=np.where(gamma>1e-8, 1/(2*2*np.pi*gamma), 0)
            elif self._dest=="M":
                # gv= np.sqrt(np.sum((self._tc._gv **2), axis=-1))[:,self._bi]
                gamma=self._tc._gamma[:,k,self._bi]
                mfp=np.where(gamma>1e-8, gv/ (2*2*np.pi*gamma), 0) # unit in nm
                self._projector=mfp
            elif self._dest=="C":#heat capacity
                V=self._tc._cell.get_volume()
                self._projector=self._tc._heat_capacity[:,k,self._bi]/V *EV/(Angstrom)**3
            if self._is_scatter:
                write_plot_scatter(self._property,
                                   self._dest,
                                   self._screen,
                                   self._projector,
                                   temp=temp,
                                   is_log= self._is_log,
                                   is_plot=self._is_plot,
                                   is_save=self._is_save)
            else:
                if self._sigma < 1e-5:
                    self.statistical_distribution()
                else:
                    self.cast_one()
                    if self._dest!="K":
                        self.set_delta_summation()
                write_distribution(self._property,
                                   self._dest,
                                   self._cast,
                                   temp=temp,
                                   sigma=self._real_sigma,
                                   is_plot=self._is_plot,
                                   is_save=self._is_save,
                                   is_cumulative=self._is_cumulative,
                                   is_log=self._is_log)

    def cast_one(self):
        if self._screen is None:
            print_error_message("The casting parameter is set incorrectly!")
        self._cast=np.zeros((self._seg, 2), dtype="double")
        if  self._dest!="K":
            desti=self._projector*self._tc._weight.reshape(-1,1)
        else:
            desti=self._projector
        pro=self._screen
        assert desti.shape==pro.shape, "Shape of %2s and shape of %2s does not match" %(self._dest, self._property)
        if self._is_fix_sigma:
            self._real_sigma = self._sigma
        else:
            self._real_sigma=self._sigma*(pro.max()-pro.min())/100
        self._cast[:,0] = allocate_space(pro, self._seg, is_log=self._is_log[0], sigma=self._real_sigma)
        if self._lang=="C":
            try:
                import _kthin as kt
                kt.distribution(pro.copy(),
                                desti.copy(),
                                self._cast,
                                self._real_sigma)
                pass
            except ImportError:
                print_warning_message("the C module _kthin is not imported, python is used instead")
                distribution_py(pro,
                                desti.copy(),
                                self._cast,
                                sigma=self._real_sigma)
        elif self._lang=="P":
            distribution_py(pro,
                            desti,
                            self._cast,
                            sigma=self._real_sigma)
        if self._is_cumulative:
            x=self._cast[:,0]
            y=self._cast[:,1]
            self._cast[0,1]=0; self._cast[1:,1] = np.cumsum((x[1:]-x[:-1]) * (y[1:]+y[:-1])/2)

    def statistical_distribution(self):
        if self._screen is None:
            print_error_message("The casting parameter is set incorrectly!")
        self._cast=np.zeros((self._seg, 2), dtype="double")
        if  self._dest!="K":
            desti=self._projector*self._tc._weight.reshape(-1,1)
        else:
            desti=self._projector
        pro=self._screen
        assert desti.shape==pro.shape, "Shape of %2s and shape of %2s does not match" %(self._dest, self._property)
        xseg = allocate_space(pro, self._seg, is_log=self._is_log[0])
        yseg = np.zeros_like(xseg)
        self._cast[:,0] = xseg
        for i in np.arange(1,self._seg):
            yseg[i] = desti[np.where((xseg[i-1] <= pro) & (pro < xseg[i]))].sum()
        if self._is_cumulative:
            self._cast[:,1] = np.cumsum(yseg)
        else:
            self._cast[:,1] = yseg

class Average():
    def __init__(self,
                 tc,
                 objects=None,
                 tool="B",
                 bi=None):
        self._tc = tc
        self._objects = objects
        self._tool = tool
        self._bi = bi

    def average(self):
        if self._tool == "B":
            normalization = self._tc._weight.sum()
            for object in self._objects:
                if object == "R":
                    gamma=self._tc._gamma[:,self._bi]
                    tau = np.where(gamma>1e-8, 1/ (2*2*np.pi*gamma), 0) # unit in ps
        elif self._tool == "K":
            for object in self._objects:
                if object == "R":
                    kappa = self._tc._kappa




