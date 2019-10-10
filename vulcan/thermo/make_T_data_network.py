import numpy as np
import scipy
import csv, ast

temp_range = {}
with open('NCHO_network_TEMPERATURE.txt') as f:
    all_lines = f.readlines()
    Rf, Rindx = {}, {} 
    i = 1
    re_tri, re_tri_k0, end_re = False, False, False
    
    for line_indx, line in enumerate(all_lines):
        
        if line.startswith("# 3-body and Disscoiation Reactions"): re_tri = True
        if line.startswith("# 3-body reactions without high-pressure rates"): re_tri_k0 = True
        
        if not line.startswith("#") and line.strip() and re_tri ==False and end_re == False:
            
            Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
            li = line.partition(']')[-1].strip()
            columns = li.split()
            Rindx[i] = int(line.partition('[')[0].strip())
            
            temp_range[Rf[i]] = columns[3]
            
            
            # print Rf[i]
#             print (columns[3])
                
        elif not line.startswith("#") and line.strip() and re_tri == True and re_tri_k0==False and end_re == False: # for 3-body
        
            Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
            li = line.partition(']')[-1].strip()
            columns = li.split()
            Rindx[i] = int(line.partition('[')[0].strip())
            
            temp_range[Rf[i]] =  columns[6] + ', ' + columns[7]
            # print (columns[6:8])
            
        elif not line.startswith("#") and line.strip() and re_tri == True and re_tri_k0==True and end_re == False: # for 3-body
        
            Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
            li = line.partition(']')[-1].strip()
            columns = li.split()
            Rindx[i] = int(line.partition('[')[0].strip())
            
            temp_range[Rf[i]] = columns[3]
            # print (columns[3])
                

with open('NCHO_photo_network_v925.txt', 'r') as f:
    ost = ''
    all_lines = f.readlines()
    Rf, Rindx = {}, {} 
    i = 1
    re_tri, re_tri_k0, special_re, photo_re, end_re = False, False, False, False, False,
    
    max_len = 0
    
    for line_indx, line in enumerate(all_lines):
        
        if line.startswith("# 3-body"): re_tri = True
        if line.startswith("# 3-body reactions without high-pressure rates"): re_tri_k0 = True
        elif line.startswith("# special"): special_re = True
        elif line.startswith("# photo"): photo_re = True
        elif line.startswith("# re_end"): end_re = True                    
        
        if not line.startswith("#") and line.strip() and re_tri==False and special_re == False and photo_re == False and end_re == False: # if not starts
                        
            Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
            after_re = line.partition(']')[-1]
            abc = after_re.split()
            
            after_re =  "{:10.2E}".format(float(abc[0])) + "{:10.3f}".format(float(abc[1])) + "{:10.1f}".format(float(abc[2])) + "{:>24}".format( after_re[26:].partition(abc[2])[-1] )
            
            # print ( "{:10.2E}".format(float(abc[0])) + "{:10.3f}".format(float(abc[1])) + "{:10.1f}".format(float(abc[2])) + after_re[after_re.index(abc[2])+1:] )
            #print (after_re[after_re.index(abc[2])+1:])
            

            li = line.partition(']')[-1].strip()
            columns = li.split()
            Rindx[i] = int(line.partition('[')[0].strip())
            
            if Rf[i] in temp_range.keys():
                #line += '{:>4}'.format(temp_range[Rf[i]]) + '\n'
                after_re = after_re[:-1]
                after_re += '{:>8}'.format(temp_range[Rf[i]])
                
            line = '{:>4}'.format(i) + ' [ ' + Rf[i] + ']'.rjust(36-len(Rf[i])) +  after_re  + '\n'

            
            i += 2 
            
        elif not line.startswith("#") and line.strip() and re_tri==True and re_tri_k0==False: 
                        
            Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
            after_re = line.partition(']')[-1]
            abc = after_re.split()
            

            after_re =  "{:10.2E}".format(float(abc[0])) + "{:10.3f}".format(float(abc[1])) + "{:10.1f}".format(float(abc[2])) +\
            "{:14.2E}".format(float(abc[3])) + "{:10.3f}".format(float(abc[4])) + "{:10.1f}".format(float(abc[5])) +  "{:>30}".format("") # +  "{:>24}".format( after_re.partition(abc[5])[-1] )
            
            # print ( "{:10.2E}".format(float(abc[0])) + "{:10.3f}".format(float(abc[1])) + "{:10.1f}".format(float(abc[2])) + after_re[after_re.index(abc[2])+1:] )
            #print (after_re[after_re.index(abc[2])+1:])
            

            li = line.partition(']')[-1].strip()
            columns = li.split()
            Rindx[i] = int(line.partition('[')[0].strip())
            
            if Rf[i] in temp_range.keys():
                #line += '{:>4}'.format(temp_range[Rf[i]]) + '\n'
                after_re = after_re[:-1]
                after_re += '{:>8}'.format(temp_range[Rf[i]])
                
            line = '{:>4}'.format(i) + ' [ ' + Rf[i] + ']'.rjust(36-len(Rf[i])) +  after_re  + '\n'

            
            i += 2    
            
            
               
        ost += line 
            

with open('NCHO_photo_network_test.txt', 'w+') as f: f.write(ost)
        
                
                
                # # switch to 3-body and dissociation reations
#                 if line.startswith("# 3-body"):
#                     re_tri = True
#
#                 if line.startswith("# 3-body reactions without high-pressure rates"):
#                     re_tri_k0 = True
#
#                 elif line.startswith("# special"):
#                     special_re = True # switch to reactions with special forms (hard coded)
#
#                 elif line.startswith("# photo"):
#                     special_re = False # turn off reading in the special form
#                     photo_re = True
#                     var.photo_indx = i
#
#                 elif line.startswith("# re_end"):
#                     end_re = True
#
#                 elif line.startswith("# wavelength switch"):
#                     var.wavelen = ast.literal_eval(all_lines[line_indx+1])
#
#                 elif line.startswith("# branching ratio"):
#                     var.br_ratio = ast.literal_eval(all_lines[line_indx+1])
#
#                 # skip common lines and blank lines
#                 # ========================================================================================
#                 if not line.startswith("#") and line.strip() and special_re == False and photo_re == False and end_re == False: # if not starts
#
#                     Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
#                     li = line.partition(']')[-1].strip()
#                     columns = li.split()
#                     Rindx[i] = int(line.partition('[')[0].strip())
#
#                     a[i] = float(columns[0])
#                     n[i] = float(columns[1])
#                     E[i] = float(columns[2])
#
#                     # switching to trimolecular reactions (len(columns) > 3 for those with high-P limit rates)
#                     if re_tri == True and re_tri_k0 == False:
#                         a_inf[i] = float(columns[3])
#                         n_inf[i] = float(columns[4])
#                         E_inf[i] = float(columns[5])
#                         list_tri.append(i)
#
#                     if columns[-1].strip() == 'He': re_He = i
#                     elif columns[-1].strip() == 'ex1': re_CH3OH = i
#
#                     # Note: make the defaut i=i
#                     k_fun[i] = lambda temp, mm, i=i: a[i] *temp**n[i] * np.exp(-E[i]/temp)
#
#
#                     if re_tri == False:
#                         k[i] = k_fun[i](Tco, M)
#
#                     # for 3-body reactions, also calculating k_inf
#                     elif re_tri == True and len(columns)>=6:
#
#
#                         kinf_fun[i] = lambda temp, i=i: a_inf[i] *temp**n_inf[i] * np.exp(-E_inf[i]/temp)
#                         k_fun_new[i] = lambda temp, mm, i=i: (a[i] *temp**n[i] * np.exp(-E[i]/temp))/(1 + (a[i] *temp**n[i] * np.exp(-E[i]/temp))*mm/(a_inf[i] *temp**n_inf[i] * np.exp(-E_inf[i]/temp)) )
#
#                         #k[i] = k_fun_new[i](Tco, M)
#                         k_inf = a_inf[i] *Tco**n_inf[i] * np.exp(-E_inf[i]/Tco)
#                         k[i] = k_fun[i](Tco, M)
#                         k[i] = k[i]/(1 + k[i]*M/k_inf )
#
#
#                     else: # for 3-body reactions without high-pressure rates
#                         k[i] = k_fun[i](Tco, M)
#
#                     ### TEST CAPPING
#                     # k[i] = np.minimum(k[i],1.E-11)
#                     ###
#
#                     i += 2
#                     # end if not
#                  # ========================================================================================
#                 elif special_re == True and line.strip() and not line.startswith("#") and end_re == False:
#
#                     Rindx[i] = int(line.partition('[')[0].strip())
#                     Rf[i] = line.partition('[')[-1].rpartition(']')[0].strip()
#
#                     if Rf[i] == 'OH + CH3 + M -> CH3OH + M':
#                         print ('Using special form for the reaction: ' + Rf[i])
#
#                         k[i] = 1.932E3*Tco**-9.88 *np.exp(-7544./Tco) + 5.109E-11*Tco**-6.25 *np.exp(-1433./Tco)
#                         k_inf = 1.031E-10 * Tco**-0.018 *np.exp(16.74/Tco)
#                         k[i] = k[i]/(1 + k[i]*M/k_inf )
#
#                         k_fun[i] = lambda temp, mm, i=i: 1.932E3 *temp**-9.88 *np.exp(-7544./temp) + 5.109E-11*temp**-6.25 *np.exp(-1433./temp)
#                         kinf_fun[i] = lambda temp, mm, i=i: 1.031E-10 * temp**-0.018 *np.exp(16.74/temp)
#                         k_fun_new[i] = lambda temp, mm, i=i: (1.932E3 *temp**-9.88 *np.exp(-7544./temp) + 5.109E-11*temp**-6.25 *np.exp(-1433./temp))/\
#                         (1 + (1.932E3 *temp**-9.88 *np.exp(-7544./temp) + 5.109E-11*temp**-6.25 *np.exp(-1433./temp)) * mm / (1.031E-10 * temp**-0.018 *np.exp(16.74/temp)) )
#
#                     i += 2
#