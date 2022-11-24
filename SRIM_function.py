import numpy as np
import pandas as pd
import os
from periodic import ELEMENTS_DATA

#точка отсчета находится в папке SRIM/userScripts, каждая функция без "_" оттуда начинается и туда возвращается

#########################################################################################################################
############ procent_Tab чисто про обработку файла из OMERE, который нужно положить в userScripts#######################
#########################################################################################################################
def procent_Tab(file): #'cosmicIons.flx'
    """Читает файл из OMERE cosmicIons.flx и:
        [0] возвращает относительный состав энергетических потоков различных ионов
        [1] возвращает соответсвующую энергию (в MeV)"""
    f = open(file, 'r')
    flx = f.read().split('\n')[28:-4]
    f.close()
    
    energy = list()
    for i in range(4, 839):
        energy += [float(flx[i].split()[0])]
    energy = np.array([0] + energy)
    d_en = energy[1:] - energy[:-1]

    i = 0
    ion = 0
    IonColumn = range(1, 93)

    En_DifFlux = [[]] * 92
    while i != len(flx) and ion < 92:
        if flx[i].find('with atomic number')!= -1:
            ion = int(list(flx[i].split())[-1])
            i+=2
            if flx[i].find('Empty flux') == -1:
                i += 2
                flux = list()
                for j in range(835):
                    flux += [list(map(float, flx[i].split()))[1] * 86400] #/day
                    i += 1
                En_DifFlux[ion-1] = np.array(flux)
            else:
                i += 3
        else:
            i += 1

    df_flux = (pd.DataFrame(En_DifFlux, index=IonColumn).fillna(0)* d_en).transpose() #число частиц dEn*плотность потоков
    full_Flux = df_flux.sum().sum()
    df_procent = df_flux/full_Flux
    return df_procent, energy[1:]


#########################################################################################################################
############ procent_Tab чисто про обработку файла из OMERE, который нужно положить в userScripts#######################
#########################################################################################################################      
         
def _ion_in_mat(ion, mm, mat):
    """возвращает энергию (в MeV) при которой частица ion проходит границу mm в материале mat
       подфункция используется в функции transenergy, которая следит за вхождением в директорию SRModule и возвращением в первоначальную директорию,
       поэтому используется _ в названии"""
    #меняем входной файл SR.IN
    sr_example = '---Stopping/Range Input Data (Number-format: Period = Decimal Point)\n---Output File Name\n"IinAl"\n---Ion(Z), Ion Mass(u)\n5   11.009\n---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.\n0    2.702    1\n---Number of Target Elements\n 1 \n---Target Elements: (Z), Target name, Stoich, Target Mass(u)\n13   "Aluminum"              1             26.982\n---Output Stopping Units (1-8)\n 5\n---Ion Energy : E-Min(keV), E-Max(keV)\n 100    5000000\n'     
    sr_in = sr_example.split('\n')
    sr_in[2] = '"' +"Iin" + mat+ '"'
    num_mat = 14 if mat=='Si' else 13
    density_mat = sr_in[6].split()
    density_mat[1] = str(ELEMENTS_DATA[num_mat]['density'])
    sr_in[6] = ' '.join(density_mat)
    param_mat = [str(num_mat), '"' + ELEMENTS_DATA[num_mat]['name'] + '"', '1', str(round(ELEMENTS_DATA[num_mat]['atomic-mass'], 3))]
    sr_in[10] = ' '.join(param_mat)
    params_ion = sr_in[4].split()
    params_ion[0] = str(ion)
    params_ion[1] = str(round(ELEMENTS_DATA[ion]['atomic-mass'], 3))
    energy = '100 ' + str(round(ELEMENTS_DATA[ion]['atomic-mass']) * 1e+6)
    sr_in[-2] = energy
    sr_in[4] = ' '.join(params_ion)
    sr_in = '\n'.join(sr_in)
    f = open("SR.IN", "w")
    f.write(sr_in)
    f.close()
    # запуск модуля
    os.system("SRModule.exe")
    
    #чтение файла и энергий
    f = open("Iin" + mat, "r")
    out = f.read().split('\n')
    f.close()
    size_range = dict({'A': 10**-7, 'um': 10**-3, 'mm': 1, 'm': 10**3})
    size_en = dict({'keV': 10**-3, 'MeV': 1, 'GeV': 10**3})
    for line in out[26:-14]:
        tmp = line.split()
        if float(tmp[4]) * size_range[tmp[5]] > mm :
            return float(tmp[0]) * size_en[tmp[1]]
    return 0
   
def crit_energy(mm, mat):
    """возвращает список энергий (в MeV) при которых пробег в материале mat частиц равен mm. Используется:
    1. если указать mat = Al и mm=3, то возвращается список энергий для каждого иона близко к которым вообще стоит начинать запускать TRIM
    2. если указать mat = Si и mm = 0.01, список энергий после которых TRIM стоит остановить, так как для нашей локальной задачи частицы с большим пробегом неинтересны
    Функция меняет директорию на SRModule, а после выполнения основной части обратно на userScripts и записывает туда результат в .txt
    """
    os.chdir("..")
    os.chdir("SR Module")
    trans_en = [_ion_in_mat(ion, mm, mat) for ion in range(1, 93)]
    os.chdir("..")
    os.chdir("userScripts")
    
    f = open('trans_en_'+str(mm)+'_mm'+mat, "w")
    f.write(' '.join([str(el) for el in trans_en]))
    f.close()
    return np.array(trans_en)

def _ion_afterAl(ion, crit_enAl_ion, crit_enSi_ion, energyOMERE):
    """запускает TRIM.exe для ионов с энергией >0.8*crit_enAl_ion и возвращает список процент из запущенных ионов прошедших через защиту, после прохождения которые имели энергию менее crit_enSi_ion (то есть запускается 100 частиц с энергией en[i] из них вообще через защиту проходит 80 частиц, из них 20 после прохождения имеют энергию меньше crit_enSi_ion, т.е. пробег в кремнии меньше заданного 10um. -> i-ый элемент в списке 0.2)
    функция работает в директории с TRIM.exe и помещается туда внешней функцией поэтому в названии _, выход функции сохраняется в userScripst\myTRIMout"""
    
    energy = energyOMERE[energyOMERE > 0.8 * crit_enAl_ion]
    if len(energy) != 0:
        out_energy = _outTRIM(energy[0], ion)
        ions_afterAl = [(energy[0], sum(out_energy <=  crit_enSi_ion))]
        protect_cond = len(out_energy)!=0 #прошла ли хотя бы одна частица защиту?
        range_cond = min(out_energy) <= crit_enSi_ion if protect_cond else True #энергия еще не слишком большая? (если частица не прошла то энергия точно еще не слишком большая)
        i = 1
        while range_cond or energy[i] <= energyOMERE[-1]:
            out_energy = _outTRIM(energy[i], ion)
            ions_afterAl += [(energy[i], sum(out_energy <=  crit_enSi_ion)/100)] #100 ионов каждой энергии было запущено
            protect_cond = len(out_energy)!=0 #прошла ли хотя бы одна частица защиту?
            range_cond = min(out_energy) <= crit_enSi_ion if protect_cond else True #энергия еще не слишком большая? (если частица не прошла то энергия точно еще не слишком большая)
            i += 1
        
        #запись в файл результата, чтобы при прерывании общей функции не потерять инфу
        text = 'ion = {ion} after 3 mm protection Al: {number}\n'.format(**{'ion': ion, 'number': ions_afterAl})
        f = open('userScripst\myTRIMout.txt', "w")
        f.write(text)
        f.close()
        return ions_afterAl
    return False

def __changeIN(energy, ion, number, tryI=False):
    """подфункция функции _outTRIM, поэтому с __, действует внутри неё. Изменяет входной файл TRIM.IN - задает энергию иона и ион в MeV (target Алюминий 3 мм!).
    tryI - прицеливание, если """
    
    in_example = '==> SRIM-2013.00 This file controls TRIM Calculations.\nIon: Z1 ,  M1,  Energy (keV), Angle,Number,Bragg Corr,AutoSave Number.\n1 1.008 200000 0 100 1 10000\nCascades(1=No;2=Full;3=Sputt;4-5=Ions;6-7=Neutrons), Random Number Seed, Reminders\n                      1                                   0       0\nDiskfiles (0=no,1=yes): Ranges, Backscatt, Transmit, Sputtered, Collisions(1=Ion;2=Ion+Recoils), Special EXYZ.txt file\n                          0       0           1       0               0                               0\nTarget material : Number of Elements & Layers\n"Aluminium                  "       1               1\nPlotType (0-5); Plot Depths: Xmin, Xmax(Ang.) [=0 0 for Viewing Full Target]\n       0                         0           3E+07\nTarget Elements:    Z   Mass(amu)\nAtom 1 = Al =       13  26.982\nLayer   Layer Name /               Width Density    Al(13)\nNumb.   Description                (Ang) (g/cm3)    Stoich\n 1      "Layer 1"           30000000  2.702       1\n0  Target layer phases (0=Solid, 1=Gas)\n0 \nTarget Compound Corrections (Bragg)\n 1  \nIndividual target atom displacement energies (eV)\n      25\nIndividual target atom lattice binding energies (eV)\n       3\nIndividual target atom surface binding energies (eV)\n    3.36\nStopping Power Version (1=2011, 0=2011)\n 0 \n'
    
    trim_in = in_example.split('\n')
    params = trim_in[2].split()
    params[0] = str(ion)
    params[1] = str(round(ELEMENTS_DATA[ion]['atomic-mass'], 3))
    params[2] = str(int(energy) * 1000)
    params[4] = str(number)
    trim_in[2] = ' '.join(params)
    if tryI:
        out = in_example.split('\n')[6].split()
        out[0] = '1'
        out[2] = '0'
        trim_in[6] = ' '.join(out)
    trim_in = '\n'.join(trim_in)
    f = open("TRIM.IN", "w")
    f.write(trim_in)
    f.close()
    print('TRIM.IN changed: I=', ion, 'energy=', energy, 'MeV')
    return os.getcwd()
         
def _outTRIM(energy, ion):
    """возвращает список энергий (в MeV) каждой частицы прошедшей через защиту (target Алюминий 3 мм!).
    как:
    1. запускает прицеливание в changeIN tryI=True, TRIM.exe запускает три иона, если они проникают не дальше (<=2.975e+7 А) ширины защиты, то не надо запускать 100 частиц, все равно не пройдут. возвращает пустой массив
    2. если останавливаются близко к краю защиты (>2.975e+7 А) или проходят через нее, то запускаем 100 частиц. Возвращает список энергий частиц прешедших через защиту
    ДЕЙСТВУЕТ В ДИРЕКТОРИИ, ГДЕ ЛЕЖИТ TRIM.exe и вызывается другой функцией которая контролирует вход в нужную директорию и возвращение обратно, поэтому содержит _ в названии.
    """
    __changeIN(energy, ion, 3, True)
    os.system("TRIM.exe")
    f = open("SRIM Outputs\RANGE_3D.txt", "r")
    tryI = f.read().split('\n')
    f.close()    
    max_depth = 3e+7 if tryI[17] == '' else max(list(map(lambda line: float(line.split()[1]) , tryI[17:-1])))
    #сделать флаг с мин энергией
    if max_depth > 2.975e+7:
        __changeIN(energy, ion, 100)
        os.system("TRIM.exe")
        f = open("SRIM Outputs\TRIMOUT.txt", "r")
        out = f.read().split('\n')
        f.close()
        en_trans = (np.array([float('0' + line.split()[3]) for line in  out[12:-1]]))/1e+6 #энергия частиц прошедших через защиту в МеВ
        return en_trans
    else: return np.array([]) 

def procent_particles_inSi_afterAl(crit_enAl, crit_enSi, df_procent, energyOMERE):
    """входные параметры:
        * критические энергии в алюминии и кремнии crit_enAl, crit_enSi
        * процент частиц в энергетическом спектре df_procent
       energyOMERE тоже определена заранее
       выходные значения:
           [0] процент частиц вообще прошедших защиту 3мм Al
           [1] процент частиц прошедших защиту 3мм Al И имеющих пробег в Si менее 10 мкм
           [2] процент частиц имеющих пробег в Si менее 10 мкм, из тех что прошедших защиту 3мм Al"""
    
    particle3mm = sum([(df_procent[ion][energyOMERE >= crit_enAl[ion-1]]).sum() for ion in df_procent.columns])
    
    os.chdir("..")
     #вызов _outTRIM для всех ионов с энергией большей crit_enAl[ion] до тех пор пока минимальная энергия вы
    particle10um = 0
    for ion in df_procent.columns:
        crit_enAl_ion = crit_enAl[ion-1]
        crit_enSi_ion = crit_enSi[ion-1]
        after_protection = _ion_afterAl(ion, crit_enAl_ion, crit_enSi_ion, energyOMERE)
        if after_protection:
            particle10um += sum([el[1] * df_procent[ion][energyOMERE==el[0]] for el in after_protection])
        print(particle10um)
    os.chdir("userScripts")
    return particle3mm, particle10um, particle10um/particle3mm
