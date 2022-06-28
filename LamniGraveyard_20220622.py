# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:37:21 2022

@author: massey_j
"""

# rotatedCharge = np.zeros_like(self.rawCharge)
# rotatedMag = np.zeros_like(self.rawMag)
# for i in range(rotatedCharge.shape[2]): 
#     rotatedCharge[...,i] = scipy.ndimage.rotate(self.rawCharge[...,i], p['Rot'], reshape = False)
#     for j in range(rotatedMag.shape[0]): 
#         rotatedMag[j, ..., i] = scipy.ndimage.rotate(self.rawMag[j, ..., i], p['Rot'], reshape = False)
# corners = CropGUI(rotatedMag)
# p.update({'Box': np.array([min(corners.corners[:,0]), max(corners.corners[:,0]),
#        min(corners.corners[:,1]), max(corners.corners[:,1])])})
# thresh = ThresholdGUI(rotatedMag)
# p.update({'thresh': thresh.theta_slider.val})
# p.update({'thetaoffset': 0}) #for now, needs fixing


def _CheckForParamsFile(self, homedir): 
    os.chdir(homedir)
    params = glob.glob('*.csv')
    
    
    if len(params) == 0: 
        print("Parameter File from previous analysis attempts not found, initiating paramater identification")
        self.Params = 0
    else:
        print("Parameter File Found")
        with open(params[0], newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            p = {}
            for row in reader:
                
                dkeys = list(row.keys())
                dkeys.remove('temp')
                temp = {key: row[key] for key in dkeys}
                temp['thresh'] = float(temp['thresh'])
                for key in ('Rot', 'thetaoffset'):
                    temp[key] = int(temp[key])
                substr = ','
                str1 = temp['Box']
                res = [i for i in range(len(str1)) if str1.startswith(substr, i)]
                b = np.zeros(shape = 4)
                b[0] = int(str1[1:res[0]])
                b[1] = int(str1[res[0]+1:res[1]])
                b[2] = int(str1[res[1]+1:res[2]])
                b[3] = int(str1[res[2]+1:len(str1) -1])
                temp['Box'] = np.array(b, dtype = int)
                p.update({row['temp']: temp})
    self.Params = p
#     def _initRotation(self): 
        RotationalGUI(self.rawCharge)

        self.Params[self.t]['Rot'] = rotated.theta_slider.val
        self.rotatedCharge = np.zeros_like(self.rawCharge)
        self.rotatedMag = np.zeros_like(self.rawMag)
        for i in range(self.rotatedCharge.shape[2]): 
            self.rotatedCharge[...,i] = scipy.ndimage.rotate(self.rawCharge[...,i], self.Params[self.t]['Rot'], reshape = False)
            for j in range(self.rotatedMag.shape[0]): 
                self.rotatedMag[j, ..., i] = scipy.ndimage.rotate(self.rawMag[j, ..., i], self.Params[self.t]['Rot'], reshape = False)
    
    def _initCrop(self): 
        corners = CropGUI(self.rotatedMag)
        while corners.flag == False: pass
        self.Params[self.t]['Box'] = np.array([min(corners.corners[:,0]), max(corners.corners[:,0]),
                min(corners.corners[:,1]), max(corners.corners[:,1])])
    
    def _initThresh(self): 
        thresh = ThresholdGUI(self.rotatedMag)
        while thresh.flag == False: pass
        self.Params[self.t]['thresh'] = thresh.theta_slider.val
        self.Params[self.t]['thetaoffset'] = 0 #for now, needs fixing