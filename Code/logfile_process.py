from copy import deepcopy
import numpy as np
import os, glob
import shutil
from . import Tool, format_change
from rdkit import Chem
from sterimol import sterimoltools

class Logfile():
    
    def __init__(self, file_dir, mol_file_dir=None,read_title=True, freq_warning=False, simple_mod = False):
        try:
            self.file_dir = file_dir
            self.mol_file_dir = mol_file_dir
            with open(file_dir, "rt") as rf:
                filelines = rf.readlines()
                filelines = [line for line in filelines if line != ""]
            self.filelines = filelines
            self.normal_end = self.is_normal_end()
            if not self.normal_end:
                self.find_error_reason()
            self.title = self.read_title()
            self.charge, self.multiplicity = self.read_charge_multiplicity()
            if self.multiplicity == -1:
                print("It's not a logfile")
                return None
            if simple_mod:
                return None
            self.method = self.read_method()
            self.file_type = 'SPE'
            if "irc" in self.method:
                self.file_type = 'IRC'
            elif "opt=modredundant" in self.method:
                self.file_type = "OM"
            elif "readfc" in self.method or "calcfc" in self.method:
                self.file_type = "TS"
            elif "opt" in self.method:
                self.file_type = "OPT"

            self.symbol_list, self.first_atom_position = self.read_first_position()
            if self.symbol_list == None:
                print("It's a wrong file with unknown wrong")
                return None
            
            if self.file_type != "SPE":
                self.running_positions = self.read_running_position()
                if self.running_positions is not None:
                    self.running_rmsd = self.react_RMSD()
                if self.normal_end and "freq" in self.method:
                    self.unreal_freq, self.unreal_freq_matrix, self.first_unreal_freq = self.read_unreal_freq(freq_warning)
                else:
                    self.unreal_freq = -1
                    self.unreal_freq_matrix = []
            else: 
                self.running_positions, self.unreal_freq, self.unreal_freq_matrix = [], 0, []
            
            if self.file_type == "OM":
                self.freeze, self.difreeze = self.read_freeze()
            else:
                self.freeze, self.difreeze = [], []


            self.all_engs, self.opt_engs = self.read_log_eng()
            if self.normal_end:
                self.running_time = self.read_log_time()
            else:
                self.engs = []
                self.running_time = 0
            if self.mol_file_dir != None and self.file_type != "SPE":
                self.bond_attach = self.check_bond_attach()
            else:
                self.bond_attach = True

            if self.file_type == "IRC":
                self.irc_result = self.irc_check()

            if self.file_type == "TS":
                if self.unreal_freq != 1:
                    print("%s is not a right TS for unreal freq num of %d" % (self.file_dir, self.unreal_freq) )
                    self.is_right_ts = False
                else:
                    self.is_right_ts = self.check_om(False)
        except:
            print("Error in %s" % self.file_dir)
            return None

        

        

    # def __repr__(self):
    #     print("Logfile Name: %s" % self.file_dir)
    #     print("Run Successful %s" % bool(self.normal_end))
    #     print("Logfile Type as %s" % self.file_type)
    #     print("Title: %s" % ' '.join([str(each) for each in self.title]))
    #     print("Charge: %d, Multiplicity: %d" % (self.charge, self.multiplicity))
    #     print("Symbol list:%s" % ' '.join(self.symbol_list))
    #     print("Number of Unreal Freq: %d" % self.unreal_freq )
    #     print("Method: %s" % self.method)
    #     print("Optimization Steps: %d" % (len(self.running_positions) - 1))
    #     print("Optimization Engs: %s" % ' '.join([str(each) for each in self.opt_engs]))
    #     print("Final Engs: %s" % ' '.join([str(each) for each in self.all_engs]))
    #     print("Elapse Time: %.3f min" % self.running_time)

    #     if self.file_type == 'IRC':
    #         print("IRC File Successful: %s" % bool(self.irc_result))
    #     if self.file_type == 'TS':
    #         print("TS File is Right: %s" % bool(self.is_right_ts))
    #     return "End"

    def is_normal_end(self):
        """Detect "Normal termination of Gaussian"

        Args:
            file_dir (str): filedir

        Returns:
            bool: whether it end normal
        """    
        lastline = self.filelines[-1]
        if lastline.find(" Normal termination of Gaussian") == -1:
            print("%s didn't run successful" % self.file_dir)
            return False
        else: return True

    def read_title(self):
        
        allfile = "".join(self.filelines)
        title = ""
        charge = 0
        # title = $$$$Title####charge????
        if allfile.find("$$$$") == -1 or allfile.find("####") == -1 or allfile.find("????") == -1:
            title = ""
            print("Can't find title")
        else:
            title = allfile[allfile.find("$$$$") + 4: allfile.find("####")]
            charge = allfile[allfile.find("####") + 4: allfile.find("????")]
            title = title.split()
            if len(title) == 6 or len(title) == 5:
                title = [int(each) for each in title]
        return title

    def read_charge_multiplicity(self):

        line_id, line = Tool.find_first_line(self.filelines, 'Charge = ', 'in')
        if line_id is None:
            print("Can't find charge and multiplicity")
            return 0, -1
        charge = int(line.split()[2])
        multiplicity = int(line.split()[-1])
        return charge, multiplicity

    def read_first_position(self):
        start_id = Tool.find_first_line(self.filelines, "Symbolic Z-matrix:", 'in')[0]
        if start_id != None:
            start_index = start_id + 2
            end_index = Tool.find_first_line(self.filelines[start_index:], ' \n', 'all')[0] + start_index
            if end_index is None:
                print("%s didn't have structure" % self.file_dir)
                return None, None
            # start_index = [i for i, line in enumerate(self.filelines) if line.find("Symbolic Z-matrix:") >= 0][0] + 2
            # end_index = [i + start_index for i, line in enumerate(self.filelines[start_index:]) if line == ' \n'][0]
            orientation = [line.split() for line in self.filelines[start_index: end_index]]
            symbol_list, position = [], []
            for each in orientation:
                if len(each)!= 4: return None, None
                symbol_list.append(each[0])
                position.append([float(each[1]), float(each[2]), float(each[3])]) 
            assert len(symbol_list) == len(position) 
            return symbol_list, np.array(position) 
        else:
            try:
                symbol, positions = self.read_running_position(read_first=1)
                return symbol, positions
            except:
                print("%s didn't have structure" % self.file_dir)
                return None, None
            


    def read_running_position(self, read_first=False):
        if self.file_type == 'IRC' or read_first:
            orientation_sign = "Input orientation:"
        else:
            orientation_sign = "Standard orientation:"
        start_indexs = [i for i, line in enumerate(self.filelines) if line.find(orientation_sign) >= 0]
        if len(start_indexs) == 0:
            print("%s Even not Input Structure, wrong file maybe" % self.file_dir)
            return None
        all_positions = []
        for each_start_index in start_indexs:
            start_index = each_start_index + 5
            end_index = Tool.find_first_line(self.filelines[start_index:], ' ---', 'in')[0] + start_index
            if end_index is None:
                break
            orientation = [line.strip("\n").split() for line in self.filelines[start_index: end_index]]
            position = []
            symbol = []
            for each in orientation:
                if len(each) != 6: return all_positions
                position.append([float(each[3]), float(each[4]), float(each[5])])
                symbol.append(int(each[1]))
            all_positions.append(position)
            
        if read_first:
            return symbol, all_positions[0]
        return np.array(all_positions)

    def read_method(self):
        method_id, method = Tool.find_first_line(self.filelines, ' #p', 'start')
        if method is None:
            print("%s with not method" % self.file_dir)
            return None
        method_final_line_id, _ = Tool.find_first_line(self.filelines[method_id:], " -------", 'start')
        method = "".join([each.strip("\n") for each in self.filelines[method_id:method_id + method_final_line_id]])
        method = method.split('#p ')[-1]
        return method

    def read_unreal_freq(self, freq_warning=True):
        """Detect unreal frequence in last positions

        Args:
            file_dir (_type_): _description_

        Returns:
            int: num of unreal freqences
        """    
        start_index = [i for i, line in enumerate(self.filelines) if '(negative Signs)' in line]
        if len(start_index) != 0: start_index = start_index[-1]
        else: start_index = 0
        smallest_freq_index, smallest_freq_line = Tool.find_first_line(self.filelines[start_index:], ' Frequencies --', "start")
        smallest_freq_index += start_index
        if smallest_freq_index == None:
            print("%s didn't calc freq" % self.file_dir)
            return -1, []
        smallest_freq_list = smallest_freq_line.strip("\n").split()[2:]
        num_unreal_freq = sum([1 for each in smallest_freq_list if float(each) < 0])
        # freq_fileline = [[lid, Tool.remove_space(line.strip("\n"))[2]] for lid, line in enumerate(self.filelines) if line.startswith(" Frequencies --")][0]
        matrix = []
        start_id = smallest_freq_index + 5
        while(1):
            line = self.filelines[start_id].strip("\n").split()
            if len(line) < 5:
                break
            line = [float(each) for each in line[2:5]]
            matrix.append(line)
            start_id += 1
        if freq_warning:
            print("%s have unreal freq" % self.file_dir)
        return num_unreal_freq, matrix, smallest_freq_list[0]

    def read_log_eng(self): 
        """read gaussian output file 

        Args:
            gjffile (str): *.log

        Returns:
            ee, zpc, cor_Energy, cor_Enthalpies, cor_Gibbs : [list with 电子能，零点能，常温热能矫正，焓矫正，自由能矫正]
        """
        all_engs = []
        opt_engs = []
        start_indexs = [i for i, line in enumerate(self.filelines) if line.find("Standard orientation: ") >= 0]
        if len(start_indexs) == 0:
            print("%s, can't find any engs" % self.file_dir)
            return all_engs, opt_engs
        for start_index in start_indexs:
            ee_line = Tool.find_first_line(self.filelines[start_index:], " SCF Done: ", "start")[-1]
            # ee_line = [line for i, line in enumerate(
            #     self.filelines[start_index:]) if line.startswith(" SCF Done: ")][-1]
            if ee_line == None: 
                ee = -1
            else:
                ee = float(ee_line.strip("\n").split()[4])
            opt_engs.append(ee)
        try:
            zpc_line = Tool.find_first_line(self.filelines[start_indexs[-1]:], " Zero-point correction=", "in")[-1]
            zpc = zpc_line.strip("\n").split(" ")[-2]
            cor_ee = Tool.find_first_line(self.filelines[start_indexs[-1]:], " Thermal correction to Energy=", "in")[-1].strip("\n").split(" ")[-1]
            cor_Enthalpies = Tool.find_first_line(self.filelines[start_indexs[-1]:], " Thermal correction to Enthalpy=", "in")[-1].strip("\n").split(" ")[-1]
            cor_Gibbs = Tool.find_first_line(self.filelines[start_indexs[-1]:], " Thermal correction to Gibbs Free Energy=", "in")[-1].strip("\n").split(" ")[-1]
            all_engs = [ee, zpc, cor_ee, cor_Enthalpies, cor_Gibbs]
            all_engs = [float(each) for each in all_engs]
        except:
            return [ee], opt_engs
        return all_engs, opt_engs

    def read_log_time(self):
        """return times (minutes)

        Args:
            gjffile (_type_): _description_

        Returns:
            _type_: _description_
        """    
        alltime = 0
        for each in self.filelines:
            if each.startswith(" Elapsed time:"):
                times = list(each.strip("\n").split())
                alltime += float(times[2]) * 24 * 60 + float(times[4]) * 60 + float(times[6]) + float(times[8]) / 60
        return alltime

    def read_freeze(self):
        freeze_start_line_id, _ = Tool.find_first_line(self.filelines, "The following ModRedundant", "in")
        freeze, difreeze = [], []
        for line in self.filelines[freeze_start_line_id + 1:]:
            line = line.strip('\n').split()
            if len(line) < 4:
                break
            if line[0] == "B":
                freeze.append([int(line[1]), int(line[2])])
            elif line[0] == "D":
                difreeze.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
        return freeze, difreeze

    def find_error_reason(self):
        errorline_id = [i for i, line in enumerate(self.filelines) if " Error termination" in line]
        # errorline_id = Tool.find_first_line(self.filelines,"start")[0]
        if len(errorline_id) == 0 and not self.normal_end:
            print(self.file_dir, "应该是没跑完")
            error_reason_line = "unfinished"
        else:
            errorline_id = errorline_id[-1]
            error_reason_line = self.filelines[errorline_id - 1]
            error_reason_line = error_reason_line.strip('.\n')
        self.error_reason = error_reason_line
    
    def solve_error_logfile(self, new_log_dir, move_file=True, savechk=None, readchk=None):
        # 选取能量最低结构，用相同方法继续跑
        # 适用于link 9999, 
        reason = self.error_reason
        if not os.path.isdir(new_log_dir):
            os.mkdir(new_log_dir)
        new_gjf_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1].split(".")[0] + '.gjf'
        if move_file:
            new_log_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1]
            shutil.move(self.file_dir, new_log_name)
        if self.file_type in ["SPE", "IRC"]:
            return 0
        if reason.endswith('link 9999') or reason == 'unfinished' or "Illegal unit" in reason:
            print("%s. Error Reason is link 9999 or unfinished." % self.file_dir)
            # opt_engs = deepcopy(self.opt_engs) 
            # for _ in range(len(opt_engs)):
            #     min_index =np.argmin(opt_engs)
            #     if self.mol_file_dir != None:
            #         bond_result = self.check_bond_attach('mol', conf_id=min_index)
            #         if bond_result:
            #             break
            #     else:
            #         break
            #     opt_engs[min_index] = 0
            title = " ".join(str(each) for each in self.title)
            if self.file_type == "TS":
                format_change.block_to_gjf(self.symbol_list, self.running_positions[-1], new_gjf_name, self.charge, title, self.method, freeze=self.freeze, difreeze=self.difreeze, savechk=savechk, readchk=readchk)
            else:
                format_change.block_to_gjf(self.symbol_list, self.running_positions[-1], new_gjf_name, self.charge, title, self.method, freeze=self.freeze, difreeze=self.difreeze, savechk=savechk, readchk=readchk)
        elif "FormBX" in reason or "Linear angle" in reason or "Tors failed for dihedral" in reason:
            new_position = self.solve_l103_problem()
            title = " ".join(str(each) for each in self.title)
            format_change.block_to_gjf(self.symbol_list, new_position, new_gjf_name, self.charge, title, self.method, freeze=self.freeze, difreeze=self.difreeze, savechk=savechk, readchk=readchk)

    def l103_error_idx(self):
        angle_idx = np.zeros(3)
        dihedral_idx = np.zeros((0, 4))
        for line in self.filelines[-15:-5]:
            line = line.strip('\n').strip(' ')
            if 'Bend failed for angle' in line:
                ww = line.split()
                angle_idx = np.array([ww[4], ww[6], ww[8]], dtype=int)

            elif 'Tors failed for dihedral' in line:
                ww = line.split()
                tmp_list = np.array([[ww[4], ww[6], ww[8], ww[10]]], dtype=int)
                dihedral_idx = np.append(dihedral_idx, tmp_list, axis=0)
            
            elif 'Linear angle in Tors.' in line:
                dihedral_idx = np.zeros((1, 4))
        if not dihedral_idx.shape[0]:
            dihedral_idx = np.zeros((1, 4))
        dihedral_idx = dihedral_idx.astype('int',copy=False)
        angle_idx= angle_idx.astype('int',copy=False)
        if not angle_idx.all() and dihedral_idx.all():
            tmp_idx = dihedral_idx[0]
            if dihedral_idx.shape[0] == 1:
                angle_idx = tmp_idx[1:]
            else:
                if (dihedral_idx[:, :3] == tmp_idx[:3]).all():
                    angle_idx = tmp_idx[:3]
                else:
                    angle_idx = tmp_idx[1:]
        return angle_idx, dihedral_idx
                    
    def l103_adjust(self):
        def get_Rotation_M(axial_v, theta):
            v = np.array(axial_v[:3])
            # 归一化
            u, v, w = v/np.linalg.norm(v)
            a = theta
            R_M = np.array([[u**2+(1-u**2)*np.cos(a),       u*v*(1-np.cos(a))-w*np.sin(a),  u*w*(1-np.cos(a))+v*np.sin(a),  0],
                            [u*v*(1-np.cos(a))+w*np.sin(a), v**2+(1-v**2) *
                            np.cos(a),        v*w*(1-np.cos(a))-u*np.sin(a),  0],
                            [u*w*(1-np.cos(a))-v*np.sin(a), v*w*(1-np.cos(a)) +
                            u*np.sin(a),  w**2+(1-w**2)*np.cos(a),        0],
                            [0,                             0,                              0,                              1]])
            return R_M

        if (self.dihedral_idx[:, :3] == self.angle_idx).all():
            atom_to_be_adjusted_idx = self.angle_idx[0]-1
            o_idx = self.angle_idx[1]-1
            v_idx = self.angle_idx[2]-1
        elif not self.dihedral_idx.all() or (self.dihedral_idx[:, 1:] == self.angle_idx).all():
            atom_to_be_adjusted_idx = self.angle_idx[-1]-1
            o_idx = self.angle_idx[-2]-1
            v_idx = self.angle_idx[-3]-1
        elif ((self.dihedral_idx[:, :3] == self.angle_idx)+(self.dihedral_idx[:, 1:] == self.angle_idx)).all():
            atom_to_be_adjusted_idx = self.angle_idx[-1]-1
            o_idx = self.angle_idx[-2]-1
            v_idx = self.angle_idx[-3]-1
        else:
            atom_to_be_adjusted_idx = self.angle_idx[-1]-1
            o_idx = self.angle_idx[-2]-1
            v_idx = self.angle_idx[-3]-1
        Coord = deepcopy(self.running_positions[-1])
        symbol_list = self.symbol_list
        angle_v = Coord[v_idx] - Coord[o_idx]
        changing_v = Coord[atom_to_be_adjusted_idx] - Coord[o_idx]
        axial_v = np.cross(angle_v, changing_v)
        axial_v = axial_v/np.linalg.norm(axial_v)
        if symbol_list[atom_to_be_adjusted_idx] == 'H':
            theta = np.pi/6.
        else:
            theta = np.pi/36.
    
        R_M = get_Rotation_M(axial_v, theta)
        o_coord = Coord[o_idx]
        tmp_v = np.append(changing_v, [1])
        tmp_v = np.dot(tmp_v, R_M)
        tmp_v = np.around(np.delete(tmp_v, 3), decimals=8)
        Coord[atom_to_be_adjusted_idx] = tmp_v + o_coord
        return Coord

    def solve_l103_problem(self):        
        self.angle_idx, self.dihedral_idx = self.l103_error_idx()
        if self.angle_idx.all():
            new_position = self.l103_adjust()
        else:
            print('!!! Warning file: %s; Unknown Error：%s'%(self.file_dir, self.error_reason))
            new_position = self.running_positions[-2]
        return new_position
        


    
    def check_bond_attach(self, standard_file='mol', print_num = False, conf_id=-1):
        assert standard_file == "mol"
        if self.running_positions is None:
            return False
        if standard_file == "mol":
            mol = Chem.MolFromMolFile(self.mol_file_dir, removeHs=False)
            position = mol.GetConformer(0).GetPositions()
            for atom_id, atom in enumerate(mol.GetAtoms()):
                if atom.GetSymbol() != self.symbol_list[atom_id] and atom.GetAtomicNum() != self.symbol_list[atom_id]:
                    print("wrong with", atom.GetIdx(), atom.GetAtomicNum(), atom_id, self.symbol_list[atom_id])
                    return False
        else:
            position = self.first_atom_position
        new_position = self.running_positions[conf_id]
        except_idxs = []
        if self.file_type == 'TS' or self.file_type == "OM":
            title = self.title
            except_idxs=[[title[0], title[2] + title[4]], [title[1], title[3] + title[4]]]
        for bond in mol.GetBonds():
            ignore=False
            start_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()
            distance_a = Tool.get_atoms_distance(position[start_atom_id], position[end_atom_id])
            distance_b = Tool.get_atoms_distance(new_position[start_atom_id], new_position[end_atom_id])
            num = distance_a / distance_b
            for except_idx in except_idxs:
                if start_atom_id in except_idx and end_atom_id in except_idx:
                    ignore=True
                    if distance_b >= 4.0 or distance_b <= 1.6:
                        print(os.path.split(self.file_dir)[-1], start_atom_id, end_atom_id, "atom may ircorrect", distance_b)
                        return False
            if print_num:
                print("%d %d %.5f" % (start_atom_id, end_atom_id, num))
            if (num <= 0.75 or num >= 1.3) and not ignore:
                print(os.path.split(self.file_dir)[-1], start_atom_id, end_atom_id, "with a wrong distance", num)
                return False
        return True
    
    def irc_check(self):
        title = self.title
        atom1, atom2, atom3, atom4, sum_ene = title[:5]
        atom3 += sum_ene
        atom4 += sum_ene
        new_position = self.running_positions[-1]
        except_idxs = [[atom1, atom3], [atom2, atom4]]
        for except_idx in except_idxs:
            start_atom_id = except_idx[0]
            end_atom_id = except_idx[1]
            distance = Tool.get_atoms_distance(new_position[start_atom_id], new_position[end_atom_id])
            if distance >= 1.7:
                print(os.path.split(self.file_dir)[-1], start_atom_id, end_atom_id, "atom may ircorrect", distance)
                return False
        return True

    def unreal_freq_improve(self, new_log_dir, savechk=None, readchk=None):
        """解决虚频问题: 将虚频振动的1.1倍带入到下一步优化结构中

        Args:
            logfile (_type_): _description_
            newdir (str, optional): _description_. Defaults to 'err_imp'.
        """    
        position = self.running_positions[-1]
        title = " ".join(str(each) for each in self.title)
        new_position = np.array(self.unreal_freq_matrix) * 1.1 + np.array(position)
        if not os.path.isdir(new_log_dir):
            os.mkdir(new_log_dir)
        new_log_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1]
        new_gjf_name = new_log_dir + "/" + os.path.split(self.file_dir)[-1].split(".")[0] + ".gjf"
        shutil.move(self.file_dir, new_log_name)
        format_change.block_to_gjf(self.symbol_list, new_position, new_gjf_name, self.charge, title, self.method, freeze=self.freeze, difreeze=self.difreeze , savechk=savechk, readchk=readchk)

    def react_RMSD(self):
        """判断优化过程中的RMSD变化

        Args:
            log_dir (_type_): _description_

        Returns:
            float: RMSD
        """    
        position_start = self.running_positions[0]
        position_end = self.running_positions[-1]
        sum_delta2 = 0
        for each_start, each_end in zip(position_start, position_end):
            delta = each_start - each_end
            sum_delta2 += delta @ delta
        sum_delta2 /= len(position_start)
        return np.sqrt(sum_delta2)

    def check_om(self, return_value=False, set_num=0.4):
        """读取过渡态结构，判断过渡态虚频是否对应反应位点

        Args:
            file_dir (str): _description_
            assert_title (_type_, optional): whether define a title. Defaults to None.

        Returns:
            Bool: 
        """    
        title = self.title
        position = deepcopy(self.running_positions[-1])
        new_position = deepcopy(self.unreal_freq_matrix)
        assert self.unreal_freq != 0
        # return new_position, position
        new_position += position
        bond1_dist_change = np.abs(Tool.get_atoms_distance(new_position[title[0]], new_position[title[2] + title[4]]) - Tool.get_atoms_distance(position[title[0]], position[title[2] + title[4]]))
        bond2_dist_change = np.abs(Tool.get_atoms_distance(new_position[title[1]], new_position[title[3] + title[4]]) - Tool.get_atoms_distance(position[title[1]], position[title[3] + title[4]]))
        if bond1_dist_change > set_num and bond2_dist_change > set_num:
            if return_value:
                return (bond1_dist_change, bond2_dist_change)
            else:
                return 1
        else:
            pass
            # print("%s may find wrong TS" % self.file_dir)
        return 0
    
    def read_orbit_eng(self, HOMO_index = [-2, -1], LUMO_index=[0,1]):
        occ_eng = []
        virt_eng = []
        lines = self.filelines
        start_id = [id for id, line in enumerate(lines) if line.startswith(" The electronic state is")][-1] + 1
        end_id = [id for id, line in enumerate(lines[start_id:]) if line.startswith("          Condensed to atoms (all electrons):")][-1] + start_id
        occ_orbits = [line for line in lines[start_id:end_id] if line.startswith(" Alpha  occ. eigenvalues")]
        virt_orbits = [line for line in lines[start_id:end_id] if line.startswith(" Alpha virt. eigenvalues")]
        for occ_orbit in occ_orbits:
            occ_eng += [float(each) * 627.5 for each in occ_orbit.strip("\n").split("--")[-1].split()]
        for virt_orbit in virt_orbits:
            virt_eng += [float(each) * 627.5 for each in virt_orbit.strip("\n").split("--")[-1].split()]

        return [occ_eng[each] for each in HOMO_index] + [virt_eng[each] for each in LUMO_index]

    def read_charge(self):
        lines = self.filelines
        start_id = [id for id, line in enumerate(lines) if line.startswith(" Mulliken charges:")][-1] + 2
        end_id = [id for id, line in enumerate(lines[start_id:]) if line.startswith(" Sum of Mulliken charges =")][-1] + start_id
        charges = []
        for line in lines[start_id:end_id]:
            charges.append(line.strip("\n").split()[-1])
            charges = [float(each) for each in charges]
        return charges
    
    def get_dipole(self):
        lines = self.filelines
        dipole_line_id = [line_id for line_id, line in enumerate(lines) if line.startswith(" Dipole moment ")]
        if len(dipole_line_id) == 0:
            print(self.filelines, "did't have a dipole moment")
            return -1

        dipole_line = lines[dipole_line_id[-1] + 1]
        dipole_moment = float(dipole_line.strip("\n").split()[-1])
        return dipole_moment

    def get_sterimol_parameters(self, atoma, atomb):
        """Calculate sterimol, atomb => substituents
        """        
        file_Params = sterimoltools.calcSterimol(file=self.file_dir, radii="cpk", atomA=atoma + 1, atomB=atomb + 1, verbose=False)
        L = file_Params.lval
        B1 = file_Params.B1
        B5 = file_Params.newB5
        return L, B1, B5

if __name__ == "__main__":
    pwd = os.getcwd()
    log_files = glob.glob(pwd + "/*.log")
    improve_dir = pwd + "/improve"
    for log_file in log_files:
        opt_log = Logfile(log_file)
        if not opt_log.normal_end:
            opt_log.solve_error_logfile(improve_dir)
        elif opt_log.unreal_freq:
            opt_log.unreal_freq_improve(improve_dir)