__author__ = 'xinjiang'
import h5py
import sys
from phonopy.units import Kb, THzToEv, EV, THz, Angstrom
from phonopy.structure.symmetry import Symmetry
from settings import  print_error_message, print_warning_message
import phonopy.structure.spglib as spg
from anharmonic.phonon3.conductivity_RTA import get_pointgroup_operations, unit_to_WmK
from phonopy.phonon.group_velocity import degenerate_sets
from phonopy.harmonic.force_constants import similarity_transformation
from anharmonic.phonon3.triplets import  get_grid_point_from_address
from phonopy.structure.spglib import get_mappings
from group_velocity import get_frame
import os
import pickle
gv_unit= (THz * Angstrom)
import numpy as np
np.seterr(divide="ignore")

def get_kpoint_group(mesh, point_operations):
    m_diag=np.diag(mesh)
    kpoint_operations= []
    for  rot in point_operations:
        rot_transform = np.dot(m_diag, np.dot(rot, np.linalg.inv(m_diag)))
        if np.allclose(rot_transform, np.rint(rot_transform)):
            kpoint_operations.append( np.rint(rot_transform) )
    return np.array(kpoint_operations, dtype=np.int)

class Kbulk():
    def __init__(self,
                 filename,
                 ts=None,
                 cutoff_lifetime=1e-4,# in second
                 cell=None,
                 mesh=None,
                 iso_file=None,
                 volume_factor=1.0,
                 nosym = False,
                 is_cahill_pohl=False):
        self._ts=ts
        self._cell=cell
        self._volume = np.abs(self._cell.get_volume()) / volume_factor
        self.set_frame()
        self._point_operations=None
        self._conversion_factor=unit_to_WmK/self._volume
        self._frequency=None
        self._gamma=None
        self._kappa=None
        self._mesh=None
        self._weight=None
        self._heat_capacity=None
        self._gamma_N=None
        self._gamma_U=None
        self._gv=None
        self._natom=None
        self._is_cahill_pohl=is_cahill_pohl
        self._cutoff_lifetime=cutoff_lifetime
        self._mapping=None
        self._map_to_ir_index=None
        self._grid=None
        self._reverse_mapping=None
        self._gv2_tensors=[]
        self._kpt_rotations_at_stars=None
        self.read(filename, mesh=mesh, vfactor = volume_factor)
        self._iso_gamma=None
        self.set_iso_gamma(iso_file)
        self.set_grid()
        self._nosym = nosym

    def set_frame(self):
        cell=self._cell.get_cell()
        self._frame, self._a =get_frame(cell)

    def read(self, filename, mesh=None, vfactor=1.0): #volume factor is used to determine the actual volume of 2D materials like graphene
        r=h5py.File(filename)
        all_ts=r['temperature'].value
        if self._ts==None:
            self._ts=all_ts
        for t in self._ts:
            if not t in all_ts:
                print_error_message("The temperatures you have chosen are beyond the range")
        t_pos=np.where(self._ts.reshape(-1,1)==all_ts)[1]
        self._frequency=r['frequency'].value
        self._natom=self._frequency.shape[-1]/3
        assert self._natom == self._cell.get_number_of_atoms(),\
            "natom in POSCAR:%d, natom in kappa file%d"%(self._natom, self._cell.get_number_of_atoms())

        self._gamma=r['gamma'].value[:,t_pos,:]
        self._kappa=r['kappa'].value[:,t_pos,:] * vfactor
        if mesh==None:
            self._mesh=np.rint([np.power(r['weight'].value.sum(), 1./3.),]*3).astype("intc")
        else:
            self._mesh=mesh
        self._heat_capacity=r['heat_capacity'].value[:,t_pos]
        self._weight=r['weight'].value

        if np.prod(self._mesh) != self._weight.sum():
            print "The mesh generated is not consistend with the one read from %s" %filename
            print "You can manually specify the mesh in the command line (--mesh)"
            sys.exit(1)
        rec_lat = np.linalg.inv(self._frame)
        qs=r['qpoint'].value
        self._qpoints=np.dot(rec_lat, qs.T).T
        self._gv=r['group_velocity'].value
        if "gamma_N" in r.keys() and "gamma_U" in r.keys():
            self._gamma_N=r['gamma_N'].value[:,t_pos,:]
            self._gamma_U=r['gamma_U'].value[:,t_pos,:]
        r.close()

    def set_iso_gamma(self, filename):
        if filename is not None:
            self._iso_gamma=np.zeros_like(self._gamma)
            f=h5py.File(filename)
            if "temperature" in f.keys():
                all_ts = f['temperature'][:]
                for t in self._ts:
                    if not t in all_ts:
                        print_error_message("The temperature %f you specified is not commensurate with those in the iso_gamma file"%t)
                t_pos=np.where(self._ts.reshape(-1,1)==all_ts)[1]
                gamma_iso=f['gamma_iso'].value[:,t_pos,:]
                assert gamma_iso.shape[0] == self._frequency.shape[0], "The gamma array in the isotope file should be the same as the one in kappa file"
                self._iso_gamma = gamma_iso
            else:
                gamma_iso = f['gamma_iso'].value
                assert gamma_iso.shape == self._frequency.shape, "The gamma array in the isotope file should be the same as the one in kappa file"
                for i, t in enumerate(self._ts):
                    self._iso_gamma[:,i,:]=gamma_iso
            f.close()

    def set_grid(self):
        self._point_operations= get_pointgroup_operations(
                    Symmetry(self._cell).get_pointgroup_operations())
        self._kpoint_operations = get_kpoint_group(self._mesh, self._point_operations)

        (mapping, rot_mappings) =get_mappings(self._mesh,
                                                    Symmetry(self._cell).get_pointgroup_operations(),
                                                    qpoints=np.array([0,0,0],dtype="double"))
        self._rot_mappings = self._kpoint_operations[rot_mappings]
        self._mapping, self._grid=spg.get_ir_reciprocal_mesh(self._mesh, self._cell)
        assert (self._mapping==mapping).all()
        self._map_to_ir_index=np.zeros_like(mapping)
        reverse_mapping=[]
        for i,u in enumerate(np.unique(mapping)):
            reverse_mapping.append(np.where(mapping==u)[0])
            self._map_to_ir_index[np.where(mapping==u)] = i
        self._reverse_mapping=reverse_mapping


    def print_kappa(self, dir=None, log_level=1):
        if dir==None:
            dir=(0,0)
        nine_dir = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1), (2,1), (2,0), (1,0)]
        ndir=nine_dir.index(dir)
        ndir -= (ndir>=6)*3 # considering the equivalence between symmetric pairs

        k=self._kappa[...,ndir].sum(axis=(0,2)) # summation over grid and branches, 0 denoting the xx direction
        print "%11s %15s %15s"%("temperature", "kappa", "Cv (J/K-m^3)")
        for i, t in enumerate(self._ts):
            cv = np.sum(self._weight[:,np.newaxis] * self._heat_capacity[:,i]) * EV / ( self._volume * Angstrom ** 3) / self._weight.sum()
            print "%10.2fK %15.5f %15.5f"%(t,k[i], cv)

    def calculate_kappa(self):
        self._kappa = np.zeros(self._gamma.shape + (6,)) #resetting self._kappa
        assert len(np.unique(self._mapping))==len(self._frequency) # irreducible qpoint number

        if self._is_cahill_pohl:
            # tau = pi/omega --> 1/(2*2*pi*gamma) = 1/(2*f) --> gamma = f/(2*pi)
            for i, t in enumerate(self._ts):
                # pos = np.where(2 * np.pi * self._gamma[:,i] < self._frequency)
                # self._gamma[:,i][pos] = self._frequency[pos] / (2 * np.pi)
                self._gamma[:,i] = self._frequency / (2 * np.pi)

        if self._iso_gamma is not None:
            self._gamma+=self._iso_gamma

        try:
            import _kthin as kthin
            num_irr = self._frequency.shape[0]
            num_band = self._frequency.shape[1]
            degeneracy = np.zeros((num_irr, num_band), dtype="intc")
            for i in range(num_irr):
                deg = degenerate_sets(self._frequency[i])
                for ele in deg:
                    for sub_ele in ele:
                        degeneracy[i, sub_ele]=ele[0]
            kthin.thermal_conductivity(self._kappa,
                                       self._rot_mappings.copy().astype("intc"),
                                       self._mapping.copy(),
                                       np.linalg.inv(self._cell.get_cell()).copy(),
                                       self._heat_capacity.copy(),
                                       self._gamma.copy(),
                                       self._gv.copy(),
                                       degeneracy.copy(),
                                       self._cutoff_lifetime)
        except ImportError:
            self.get_rotations_for_stars()
            for i, grid_point in enumerate(np.unique(self._map_to_ir_index)):
                gv2_tensors=self.get_gv_by_gv(i)
                self._gv2_tensors.append(gv2_tensors)
                # Sum all vxv at k*
                gv_sum2 =self.get_gv_sum(gv2_tensors)
                for k, l in list(np.ndindex(len(self._ts), self._natom*3)):
                    if self._gamma[i, k, l] < 0.5 / self._cutoff_lifetime / THz:
                        continue
                    self._kappa[i, k, l, :] = (
                        gv_sum2[:, l] * self._heat_capacity[i,k, l] / self._gamma[i, k, l])
        self._kappa *= self._conversion_factor / (2 * self._weight.sum())

    def get_gv_sum(self, gv2_tensors):
        gv_sum2=np.zeros((6, self._natom*3), dtype='double')
        for j, vxv in enumerate(
            ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            gv_sum2[j] = gv2_tensors[:, :, vxv[0], vxv[1]].sum(axis=0)
        return gv_sum2

    def get_gv_by_gv(self,i):
        deg_sets = degenerate_sets(self._frequency[i])
        orbits, rotations = self._reverse_mapping[i], self._kpt_rotations_at_stars[i]
        # self._get_group_veclocities_at_star(i, gv)
        gv2_tensor = []
        rec_lat = np.linalg.inv(self._cell.get_cell())
        rotations_cartesian = [similarity_transformation(rec_lat, r)
                               for r in rotations]
        for rot_c in rotations_cartesian:
            gvs_rot = np.dot(rot_c, self._gv[i].T).T
            # Take average of group veclocities of degenerate phonon modes
            # and then calculate gv x gv to preserve symmetry
            gvs = np.zeros_like(gvs_rot)
            for deg in deg_sets:
                gv_ave = gvs_rot[deg].sum(axis=0) / len(deg)
                for j in deg:
                    gvs[j] = gv_ave
            gv2_tensor.append([np.outer(gv, gv) for gv in gvs])
        return np.array(gv2_tensor)

    def get_gv_nosym(self):
        if self._kpt_rotations_at_stars == None:
            self.get_rotations_for_stars()
        num_band = self._frequency.shape[1]
        gv_nosym = np.zeros((self._weight.sum(), num_band,3), dtype="double")
        for i, freq in enumerate(self._frequency):
            deg_sets = degenerate_sets(freq)
            orbits, rotations = self._reverse_mapping[i], self._kpt_rotations_at_stars[i]
            gv_at_orbits = np.zeros((len(orbits), num_band, 3), dtype="double")
            rec_lat = np.linalg.inv(self._cell.get_cell())
            rotations_cartesian = [similarity_transformation(rec_lat, r)
                                   for r in rotations]
            for j, rot_c in enumerate(rotations_cartesian):
                gvs_rot = np.dot(rot_c, self._gv[i].T).T
                # Take average of group veclocities of degenerate phonon modes
                # and then calculate gv x gv to preserve symmetry
                gvs = np.zeros_like(gvs_rot)
                for deg in deg_sets:
                    gv_ave = gvs_rot[deg].sum(axis=0) / len(deg)
                    for k in deg:
                        gv_at_orbits[j,k] = gv_ave
            gv_nosym[orbits] = gv_at_orbits
        return gv_nosym

    def get_qpoints_nosym(self):
        if self._kpt_rotations_at_stars == None:
            self.get_rotations_for_stars()
        qpoints_nosym = np.zeros((self._weight.sum(),3), dtype=float)
        for i in np.arange(len(self._weight)):
            orbits, rotations = self._reverse_mapping[i], self._kpt_rotations_at_stars[i]
            qpoints_at_orbits = np.zeros((len(orbits), 3), dtype=float)
            for j, rot_c in enumerate(rotations):
                qpoints_at_orbits[j] = np.dot(rot_c, self._qpoints[i])
            qpoints_nosym[orbits] = qpoints_at_orbits
        return qpoints_nosym

    def get_frequency_nosym(self):
        if self._kpt_rotations_at_stars == None:
            self.get_rotations_for_stars()
        frequency_nosym = np.zeros((self._weight.sum(), self._frequency.shape[1]), dtype="double")
        for i in np.arange(len(self._weight)):
            frequency_nosym[self._reverse_mapping[i]] = self._frequency[i]
        return frequency_nosym

    def get_weight_nosym(self):
        return np.ones(self._weight.sum(), dtype=int)

    def get_kappa_nosym(self):
        if self._kpt_rotations_at_stars == None:
            self.get_rotations_for_stars()
        kappa_nosym = np.zeros((self._weight.sum(),) + self._kappa.shape[1:], dtype = "double")
        for i in np.arange(len(self._weight)):
            kappa_nosym[self._reverse_mapping[i]] = self._kappa[i] / self._weight[i]

        return kappa_nosym

    def get_gamma_nosym(self):
        if self._kpt_rotations_at_stars == None:
            self.get_rotations_for_stars()
        gamma_nosym = np.zeros(((self._weight.sum(),) + self._gamma.shape[1:]), dtype="double")
        for i in np.arange(len(self._weight)):
            gamma_nosym[self._reverse_mapping[i]] = self._gamma[i]
        return gamma_nosym

    def set_nosym(self):

        gv_nosym = self.get_gv_nosym()
        qpoints_nosym = self.get_qpoints_nosym()
        frequency_nosym = self.get_frequency_nosym()
        weight_nosym = self.get_weight_nosym()
        kappa_nosym = self.get_kappa_nosym()
        gamma_nosym = self.get_gamma_nosym()
        self._gv, self._qpoints, self._frequency, self._weight, self._gamma, self._kappa =\
            gv_nosym, qpoints_nosym, frequency_nosym, weight_nosym, gamma_nosym, kappa_nosym


    def get_rotations_for_stars(self):
        "Finding the equivalent qpoints as the original q-star and their corresponding rotation matrices"
        if os.path.exists("reverse_mapping-m%d%d%d"%tuple(self._mesh)) and os.path.exists("kpt_rotations_at_stars-m%d%d%d"%tuple(self._mesh)):
            self._reverse_mapping=pickle.load(open("reverse_mapping-m%d%d%d"%tuple(self._mesh),"rb"))
            self._kpt_rotations_at_stars=pickle.load(open("kpt_rotations_at_stars-m%d%d%d"%tuple(self._mesh),"rb"))
        else:
            try:
                kpt_rotations_at_stars=[]
                reverse_mapping=[]
                for index, grid_point in enumerate(np.unique(self._mapping)):
                    equi_pos = np.where(self._mapping == grid_point)
                    kpt_rotations_at_stars.append([np.linalg.inv(rot) for rot in self._rot_mappings[equi_pos]])
                    reverse_mapping.append(equi_pos[0])
                    assert len(kpt_rotations_at_stars[index]) == self._weight[index]

                self._kpt_rotations_at_stars = kpt_rotations_at_stars
                self._reverse_mapping = reverse_mapping

            except ImportError:
                all_orbits=[]
                kpt_rotations_at_stars=[]
                for index, grid_point in enumerate(np.unique(self._mapping)):
                    orig_address = self._grid[grid_point]
                    orbits = []
                    rotations = []
                    for rot in self._kpoint_operations:
                        rot_address = np.dot(rot, orig_address) % self._mesh
                        in_orbits = False
                        for orbit in orbits:
                            if (rot_address == orbit).all():
                                in_orbits = True
                                break
                        if not in_orbits:
                            orbits.append(rot_address)
                            rotations.append(rot)
                    # check if the number of rotations is correct.
                    assert len(rotations) == self._weight[index], \
                           "Num rotations %d, weight %d" % (
                               len(rotations), self._weight[index])
                    all_orbits.append(orbits)
                    kpt_rotations_at_stars.append(rotations)
                self._reverse_mapping=[[get_grid_point_from_address(o, self._mesh) for o in p]for p in all_orbits]
                assert len(sum(self._reverse_mapping, []))==len(self._mapping)
                self._kpt_rotations_at_stars=kpt_rotations_at_stars
                pickle.dump(self._reverse_mapping, open("reverse_mapping-m%d%d%d"%tuple(self._mesh),"wb"))
                pickle.dump(self._kpt_rotations_at_stars, open("kpt_rotations_at_stars-m%d%d%d"%tuple(self._mesh),"wb"))

class Kthin():
    def __init__(self, kbulk,
                 thick,
                 rough=None,
                 dirw=-1, # direction for the thickness (width)
                 specularity=None,
                 group_velocity=None,
                 language="C",
                 direc=None,
                 log_level=1,
                 thick_unit = None,
                 rough_unit = None):
        # self.__dict__=dict(kbulk.__dict__) # copy all the variables from kbulk to self
        self.copy(kbulk)
        self._gamma_bulk=kbulk._gamma
        self._kbulk=kbulk
        if thick_unit is not None:
            self._thick_unit = thick_unit / self._a
        else:
            self._thick_unit = 1.0
        if rough_unit is not None:
            self._rough_unit = rough_unit / self._a
        else:
            self._rough_unit = 1.0
        self.set_heat_capacity()
        if group_velocity is not None:
            self.get_gv_from(group_velocity)
        else:
            self.get_gv_nosym()
            # print_error_message("gv shift file is not provided")
        if self._weight.sum() != np.prod(kbulk._mesh):
            print "Error! Mesh from gv file is inconsistent with the one from kappa file!"
            sys.exit(1)
        self._thick=thick * self._thick_unit
        self._dirw = dirw
        self._is_rough=False
        self._rough=None
        self._log_level = log_level
        self._language=language
        if direc==None:
            self._dir=(0,0)
        else:
            self._dir=direc
        if specularity==None:
            if rough==None:
                self._specularity=np.array([0.])
            else:
                self._is_rough=True
                self._rough=rough * self._rough_unit
                self._specularity=None
        else:
            self._specularity=specularity

    def copy(self, k):
        self._ts=k._ts
        self._cell=k._cell
        self._frame = k._frame
        self._a=k._a
        self._point_operations=k._point_operations
        self._conversion_factor=k._conversion_factor
        self._frequency=k._frequency
        self._gamma=k._gamma
        self._mesh=k._mesh
        self._weight=k._weight
        self._heat_capacity=k._heat_capacity
        self._gv=k._gv
        self._natom=k._natom
        self._cutoff_lifetime=k._cutoff_lifetime
        self._mapping=k._mapping
        self._map_to_ir_index=k._map_to_ir_index
        self._grid=k._grid
        self._reverse_mapping=k._reverse_mapping
        self._gv2_tensors=k._gv2_tensors
        self._kpt_rotations_at_stars=k._kpt_rotations_at_stars
        self._iso_gamma=k._iso_gamma

    def set_heat_capacity(self):
        cv=np.zeros((self._weight.sum(), len(self._ts), 3*self._natom), dtype="double")
        for i, grid_point in enumerate(np.unique(self._map_to_ir_index)):
            cv[self._reverse_mapping[i]]=self._heat_capacity[i]
        self._heat_capacity=cv


    def get_gv_from(self, gv_class):
        self._gv=gv_class._gv
        self._weight=gv_class._weight
        self._qpoints=gv_class._qpoints
        self._frequency=gv_class._frequency

    def get_gv_nosym(self):
        self._gv = self._kbulk.get_gv_nosym()
        self._qpoints = self._kbulk.get_qpoints_nosym()
        self._frequency = self._kbulk.get_frequency_nosym()
        self._weight = self._kbulk.get_weight_nosym()

    def get_gv_by_gv(self,i):
        orbits = self._reverse_mapping[i]
        gv2_tensor=[]
        for o in orbits:
            gvs=self._gv[o]
            gv2_tensor.append([np.outer(gv,gv) for gv in gvs])
        return np.array(gv2_tensor) # dim of gv2: [equivalent_qs ,s, 3,3]

    def get_mode_wise_specularity(self, grid_index):
        return np.exp(-16 * np.pi**3 * self._rough_temp**2 *self._qpoints[grid_index, self._dirw] **2 )

    def get_scaling_factor(self, grid_index, tem_index, band_index):
        factors=[]
        thick=self._thick_temp * self._a * Angstrom
        if self._dirw in [0, 1, 2]:
            gvs = self._gv[:,:,self._dirw]
        else:
            gvs = np.sqrt(np.sum(self._gv ** 2, axis=-1))
        for o in self._reverse_mapping[grid_index]:
            gv=gvs[o, band_index] * gv_unit
            if np.abs(gv)<1e-8:
                factors.append(1)
                continue
            if self._is_rough:
                p=self.get_mode_wise_specularity(o)
            else:
                p=self._p_temp
            gamma=self._gamma_bulk[grid_index, tem_index, band_index] * THz
            delta=2*gamma*(2*np.pi) *thick/ np.abs(gv)
            factors.append(delta*(1+p)/(2*(1-p)+delta*(1+p)))
        return np.array(factors) #dim [equiv]

    def set_gamma(self):
        self._gamma=np.zeros_like(self._kappa)
        for i, grid_point in enumerate(np.unique(self._map_to_ir_index)):
            self._gamma[:,:,self._reverse_mapping[i]]=self._gamma_bulk[i]
        iso_gamma=np.zeros((self._weight.sum(), len(self._ts), 3*self._natom), dtype="double")
        if self._iso_gamma is not None:
            for i, grid_point in enumerate(np.unique(self._map_to_ir_index)):
                iso_gamma[self._reverse_mapping[i]]=self._iso_gamma[i]
        self._iso_gamma=iso_gamma

    def calculate_kappa(self):
        if self._is_rough:
            self._kappa=np.zeros((len(self._thick),
                                       len(self._rough),
                                       len(self._qpoints),
                                       len(self._ts),
                                       self._natom*3),
                                      dtype="double")
        else:
            self._kappa=np.zeros((len(self._thick),
                                  len(self._specularity),
                                  len(self._qpoints),
                                  len(self._ts),
                                  self._natom*3),
                                 dtype="double")
        self.set_gamma()

        if self._language == "C":
            try:
                import _kthin
                self.calculate_kappa_c()
            except ImportError:
                print_warning_message("the C module _kthin is not imported, python is used instead")
                self.calculate_kappa_py()
        else:
            self.calculate_kappa_py()

    def calculate_kappa_py(self):
        for t, thick in enumerate(self._thick):
            self._thick_temp=thick
            for r in np.arange(self._kappa.shape[1]): #specularity or roughness
                if self._is_rough:
                    self._rough_temp=self._rough[r]
                else:
                    self._p_temp=self._specularity[r]
                for i, grid_point in enumerate(np.unique(self._map_to_ir_index)):
                    if len(self._gv2_tensors)== 0:
                        gv2_tensors=self.get_gv_by_gv(i)
                    else:
                        gv2_tensors=self._gv2_tensors[i]
                    for k, l in list(np.ndindex(len(self._ts), self._natom*3)):
                        if self._gamma_bulk[i, k, l] < 0.5 / self._cutoff_lifetime / THz:
                            continue
                        factor=self.get_scaling_factor(i, k, l)
                        # assert len(factor) == self._weight[i]
                        self._kappa[t, r,i, k, l] = np.sum(
                            gv2_tensors[:,l, self._dir[0], self._dir[1]] * self._heat_capacity[i,k, l] / (self._gamma_bulk[i, k, l] * 2) * factor*
                            self._conversion_factor)
                        self._gamma[t,r,self._reverse_mapping[i],k,l] = np.where(factor>1e-8, self._gamma_bulk[i, k, l]/ factor,0)
                self._kappa[t,r] /= self._weight.sum()

    def calculate_kappa_c(self):
        import _kthin as kt
        if self._is_rough:
            kt.kappa_thin_film_pvary(self._a,
                                     self._kappa,
                                     self._gamma,
                                     self._iso_gamma,
                                     self._thick.astype("double"),
                                     self._rough.astype("double"),
                                     self._qpoints.copy(),
                                     self._heat_capacity.copy(),
                                     self._gv.copy(),
                                     self._cutoff_lifetime,
                                     np.array(self._dir, dtype="intc"),
                                     self._dirw)
        else:
            kt.kappa_thin_film_pconst(self._a,
                                      self._kappa,
                                      self._gamma,
                                      self._iso_gamma,
                                      self._thick.astype("double"),
                                      self._specularity.astype("double"),
                                      self._heat_capacity.copy(),
                                      self._gv.copy(),
                                      self._cutoff_lifetime,
                                      np.array(self._dir, dtype="intc"),
                                      self._dirw)
        self._kappa *= self._conversion_factor/ self._weight.sum()

    def print_kappa_thin(self):
        if self._is_rough:
            print_var = self._rough*self._a/10
            print_string="Roughness"
        else:
            print_var=self._specularity
            print_string="Specularity"
        print "%20s\t%11s\t%11s\t%15s\t%15s"%("thickness", print_string, "Temperature" , "K_thin", "k_thin/k_bulk")
        for j in np.arange(self._kappa.shape[1]):
            for k, temp in enumerate(self._ts):
                for i, thick in enumerate(self._thick):
                    kappa_thin=self._kappa[i,j, :, k].sum()
                    print "%20f\t%11f\t%11f\t%15.8f\t%15.10f" \
                          %(thick*self._a/10, print_var[j], temp, kappa_thin, kappa_thin/self._kbulk._kappa[:,k,:, 0].sum())
            #     print
            # print

    def print_kappa_branch(self):
        nb=self._natom*3 #number of branches
        if self._is_rough:
            print_var = self._rough*self._a/10
            print_string="Roughness"
        else:
            print_var=self._specularity
            print_string="Specularity"
        print "%20s\t%11s\t%11s\t%15s\t%15s"%("thickness", print_string, "Temperature" , "K_thin", "k_thin/k_bulk") + \
            "\t%8s"*nb %tuple(["b%s"%i for i in np.arange(nb)])
        for j in np.arange(self._kappa.shape[1]):
            for k, temp in enumerate(self._ts):
                for i, thick in enumerate(self._thick):
                    kappa_thin=self._kappa[i,j, :, k].sum()
                    print "%20f\t%11f\t%11f\t%15.8f\t%15.10f" \
                          %(thick*self._a/10, print_var[j], temp, kappa_thin, kappa_thin/self._kbulk._kappa[:,k,:, 0].sum()) +\
                            "\t%8.4f"*nb %tuple(self._kappa[i,j,:,k].sum(axis=0))

class Kultrathin():
    def __init__(self, kbulk, rough=None, specularity=None, group_velocity=None, language="C"):
        self.copy(kbulk)
        self._gamma_bulk=kbulk._gamma
        self._kbulk=kbulk
        self.set_heat_capacity()
        if group_velocity is not None:
            self.get_gv_from(group_velocity)
        else:
            print_error_message("gv shift file is not provided")
        self._is_rough=False
        self._rough=None
        self._language=language
        if specularity==None:
            if rough==None:
                self._specularity=np.array([0.])
            else:
                self._is_rough=True
                self._rough=rough
                self._specularity=None
        else:
            self._specularity=specularity

    def copy(self, k):
        self._ts=k._ts
        self._cell=k._cell
        self._frame = k._frame
        self._a=k._a
        self._point_operations=k._point_operations
        self._conversion_factor=k._conversion_factor
        self._frequency=k._frequency
        self._gamma=k._gamma
        self._mesh=k._mesh
        self._weight=k._weight
        self._heat_capacity=k._heat_capacity
        self._gv=k._gv
        self._natom=k._natom
        self._cutoff_lifetime=k._cutoff_lifetime
        self._mapping=k._mapping
        self._map_to_ir_index=k._map_to_ir_index
        self._grid=k._grid
        self._reverse_mapping=k._reverse_mapping
        self._gv2_tensors=k._gv2_tensors
        self._kpt_rotations_at_stars=k._kpt_rotations_at_stars
        self._iso_gamma=k._iso_gamma