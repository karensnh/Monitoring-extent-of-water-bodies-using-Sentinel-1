
#!/usr/bin/env python
# coding: utf-8

#####Script to take in sentinel 1 ARD backscatter data, generate derived layers (VV/VH ratio and VV-VH difference), generate machine learning model to distinguish water/ non water areas, classify pixels in area of interest as water/ non water
#
#
##Inputs are:
#   - sentinel 1 ARD data (VV and VH) (imagery in a folder)
#   - shapefile of training data (permanent water and permanent land)
#   - shapefile of extent of area of interest - AOI - (this is the extent of the output raster and must encompass the sites of interest and the training data)
# 
##Outputs are:
#   - classified raster for each date , for extent of AOI
#
###Folder structure labelled:
#   - output_image (for output classified rasters),
#   - training_data (includes shapefile of AOI and shapefile of training data - water/ land)
#   - temp (for intermediate raster output that gets overwritten for each date)
#   - input_images (input rasters of sentinel 1)
# 
###Main steps:
#   - extracts pixel values for polygons
#   - resamples pixels , default is 300 pixel per class (total number pixels required from each polygon are estimated based on polygon size, sampling is random where total number of pixels required is less than that in polygon, if number of pixels required is greater than number in polygon then all are sampled and addtional pixels are randomly sampled.)
#   - splits into training/ validation data (does not take into account source polygon),
#   - trains machine learning model (random forest), assesses accuracy using separate validation and training data. 
#           - if model accuracy is >=0.8, rebuilds model based on all training data(validation and training) then classifies input image and writes out:
#                -  binary raster (1== water, 0== not water)
# 
###To use a different number of bands:
# - Update the script to the new number in the the 9th code block ('extract all data for all pixels'), where code is highlighted with lines of ######, and select which bands should be included (again highlighted in this code block with lines of #########).
# - Update the final code block with band to be excluded (highlighted with lines of ####)
# 
# 

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split
import geopandas as gpd
import descartes
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from rasterstats import zonal_stats
import datetime
import math
import glob
import fiona
import rasterio.mask
from osgeo import gdal
import shutil
from rasterio.merge import merge
#import gdal_merge as gm

# set working drive, iteration (for naming outputs), number of pixels per class
wd = '/gws/nopw/j04/jncc_muirburn/users/kf_waterbodies/'
#iteration = 'v1'
samplesize = 300 #number of pixels to be sampled per class (approximate)
output_prob = "n"# "y"


# ###Set image directory, training directory
# ### Set training , and aoi filenames
# ###All images in input_images directory with tif extension will be read in 
##all inputs are in British National Grid
image_dir = os.path.join(wd, 'input_images')
temp_dir = os.path.join(wd, 'temp')

training_dir= os.path.join(wd, 'training_data')
AOI_dir = os.path.join(wd, 'training_data')
output_dir = os.path.join(wd, 'output_image')




download_dir = os.path.join(wd, 'download')# 
new_image_dir = os.path.join(wd, 'input_images')

#filenames
training_fn ="Highland_Test_Zones_class.shp"

aoi_fn = "Highland_AOI.shp"



#merge aqcuisitions from the same day
###

imagefiles =  glob.glob(os.path.join(download_dir, '*.tif'))
print(imagefiles)


image_ids = []
image_dates =[]
image_date_pairlist = [] #date:filename

for image in imagefiles:

  imagename = image[-64-31]
  imagedate = image[-60:-52]


  image_ids.append(imagename)
  image_dates.append(imagedate)
  image_date_pairlist= list(zip(image_dates, imagefiles))
#pair those with the same date, otherwise keep single
from collections import defaultdict
d= defaultdict(list)
for k,v in image_date_pairlist:
  d[k].append(v)
print(d.items())
type(d.items())

keytupule = d.keys()
print(len(keytupule))
valuetupule = d.values()

print(valuetupule)
print(len(list(valuetupule)[0]))

#iterate through dictionary, write sym links unchanged to infile_mergedimages, if more than 1 item ,
#merge these and write output to infile_mergedimages

for key, value in d.items():
  print(type(value))
  print(value)
  if len(value) == 1:

    shutil.move(d[key][0],  new_image_dir)
    pass
 #        #write to output
  else:

    filename = d[key][0][-64:]
    print("filename =")
    print(filename)
    outputfp = os.path.join(new_image_dir, filename)
    #merge the two images from single date capture, give name of first file, save to new folder
    

    src_files_to_mosaic = []
    fps = d[key]
    for fp in fps:
      src = rasterio.open(fp)
      src_files_to_mosaic.append(src)

    #src_files_to_mosaic
    mosaic, out_trans =merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans})

# Write the mosaic raster to disk
    with rasterio.open(outputfp, "w", **out_meta) as dest:
        dest.write(mosaic)

# #open polygon dataset
from pyproj import CRS

crs=CRS('EPSG:27700')


TaggedPolys= gpd.read_file(os.path.join(training_dir, training_fn))

# #create list of input images


imagefiles =  glob.glob(os.path.join(image_dir, '*.tif'))
print(imagefiles)


image_ids = []

for image in imagefiles:
  imagename = image[-68:-31]
  image_ids.append(imagename)
print(image_ids)


# define function to extract all data for all pixels where the centroid falls within polygons, add polygon id and category id to final 2 columns




Binitial=2
def getPixels(image, poly, indexInput, polygons,  target):
    global Binitial
    shape=[mapping(poly)] 
    print((shape))
    print(("got to -1"))

    outImage, out_transform = rasterio.mask.mask(image, shape, crop=True, nodata=np.nan)#reduce imagery to pixels overlapping polygon

    # if nodata is set to a figure pixels where centroid is outwith polygon are included and not excluded with drop.na
    outList = outImage.reshape((Binitial, -1)).T# reshape output array to rows equal to number of bands, and number of columns to match input (-1) 
    
    currentPolyID = polygons.loc[indexInput,"Poly_ID"]# get current polygon ID
   
    currentPolyIDarr= np.repeat(currentPolyID, outList.shape[0])# creates 1D array of polyID, size equal to number of pixels (shape returns rows, columns)
    currentPolyIDarr= currentPolyIDarr.reshape((outList.shape[0],1))# creates 2D array, 1column
    currentCategory =polygons.loc[indexInput,"Water"] #ID
    
    currentCategoryarr= np.repeat(currentCategory, outList.shape[0])
    currentCategoryarr= currentCategoryarr.reshape((outList.shape[0],1))# create 2D array of current class / category
    
    outList = np.concatenate((outList,currentPolyIDarr), axis = 1)# add poly ID to pixel values
    outList = np.append(outList,currentCategoryarr, axis=1)# add class to pixel values
    outList = pd.DataFrame(outList).dropna()
 
    return np.append(target, outList, axis=0)


def extractAllPolygons(image, featuresgeom, features):
    global Binitial # number of bands in input imagt
    finalcolno = Binitial+2 # number of columns in extracted pixel dataset
    flatten = np.array([]).reshape(0,finalcolno).astype(float)# empty dataset with number of colums set and datatype set to float
    for index, f in enumerate(featuresgeom): #iterate through each polygon
      indexInput= index# iteration number
      flatten = getPixels(image,f,indexInput, features, flatten)
    flattenArr = np.ma.masked_array(flatten, mask=(flatten == np.nan))
    return pd.DataFrame(flattenArr).dropna()# remove any na - machine learning models can't deal with them 


# In[ ]:


##for each raster in the input dir, create extra 2 bands, creat ML model, apply model, write output
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import tempfile
B=4 # number of bands including 2 derived , calculated for extracted pixels only

for image in imagefiles:
  imagename = image[-68:-31]
  s1 = rasterio.open(image)

  #assume sections small enough not to require windowed reading

  VV = s1.read(1).astype(float)
  VH = s1.read(2).astype(float)
  print(type(VV))
  
  
  VVshape = VV.shape
  print("VV shape, VV[0],VV[1],VV[2]")
  print((VVshape))
  print(VV[0])
  print(VVshape[1])
  print((VV[200,0:50]))
  print((VH[200,0:50]))
  np.seterr(divide='ignore', invalid = 'ignore' ) #ignore errors from calculating indices/ ratios#over= 'ignore', under = 'ignore' 
  #Copy raster profile to for later output
  s1prof = s1.profile.copy()
  s1prof.update(count = 1, nodata=None, dtype=np.float64)

  #extract pixel values from training data
  totValues = extractAllPolygons(s1,TaggedPolys.geometry.values, TaggedPolys)# runs 2 functions, input is imagery, geometry part of pandas array, gp dataframe


  #insert 2 columns and calculate VV:VH ratio and VV VH difference
  newcolVVVHratio = pd.DataFrame(totValues[0]/totValues[1])
  newcolVVVHdiff = pd.DataFrame(totValues[0]-totValues[1])

  #create new pd dataframe including VVVh ratio and VVVH diff while preserving indices of colums (0,1,2,3,4,5)
  ##### use concat to keep indexing 
  newdf = pd.concat([totValues.iloc[:,[0,1]],newcolVVVHratio, newcolVVVHdiff, totValues.iloc[:,[2,3]]], ignore_index=True, axis =1)
  totValues = newdf


  ####select appropriate number of pixels for each polygon, based on size of polygon and total size of area for each class. 
  ####Select all data and resample where required number of pixels exceeds total pixels in polygon, where number of pixels 
  ####required is less than that in polygon take random selection.
  ClassColumnIndex = 5 # get column index for class ID
  PolygonColumnIndex = 4 # get column index for polygon ID
  Classes = totValues.iloc[:,ClassColumnIndex].unique()


  #iloc is index rather than name
  FinValues = pd.DataFrame()#create empty pandas dataframe
  for Class in Classes: #iterate through each class
    ClassValues = totValues[totValues.iloc[:,ClassColumnIndex]==Class]# select all rows from totValues for this class 
    totpixels = ClassValues.shape[0] # get number of rows (equals number of pixels) for this class from all training polygons
    print((' class :{}  tot pixels = {}'. format(Class,totpixels) ))

    ClassPolygonIDs = ClassValues.iloc[:,B].unique() #obtain polyIDs for this class
    for polyID in ClassPolygonIDs: #iterate through polygons, sample from each polygon based on size
      ClassPolygonValues = ClassValues[ClassValues.iloc[:,B]==polyID]# pixel values for this class , this polygon
      PolySize = ClassPolygonValues.shape[0] # get number of rows (equals number of pixels) for this polygon
      ReqPixels = ((int(math.ceil((PolySize/totpixels)*samplesize)))) #number of pixels required from this polygon, taking into account overall number in training data and total pixels required for class
      print(("Total pixels for poly = {}".format(PolySize)))
      print(("Number of pixels for class {}, polyId {} is :{}".format(Class, polyID,ReqPixels )))

      if PolySize>= ReqPixels:# random selection if number of pixels in polygon is greater than number required
        SelectedValues = ClassPolygonValues.sample(ReqPixels, replace=False)
        
        FinValues = FinValues.append(SelectedValues)
        
      elif PolySize < ReqPixels:# select all pixels and then resample with replacement if number of pixels in polygon is lower than number required
        Extrasamplesize = ReqPixels - PolySize
        ExtraValues = ClassPolygonValues.sample(Extrasamplesize, replace=True)

        SelectedValues = pd.concat((ExtraValues,ClassPolygonValues), axis = 0)
        FinValues = FinValues.append(SelectedValues)
 

  ################ split into training and test data predictor and output class ########################
  trainX =pd.DataFrame(columns=range(4))
  testX = pd.DataFrame(columns=range(4))
  trainy = pd.DataFrame(columns=range(1))
  testy = pd.DataFrame(columns=range(1))


  for Class in Classes:
    ClassSampledValues = FinValues[FinValues.iloc[:,ClassColumnIndex]==Class ] #select current class from all pixel values
    ClassSampledValuesTrain, ClassSampledValuesTest = train_test_split(ClassSampledValues, test_size = 0.2, random_state = 999 )

    trainXClass = ClassSampledValuesTrain.iloc[:,0:B]#select training data colums
    #selects columns from index 0 up to but not including the number of bands - so if you start counting from 1 it selects up to  column B
    trainyClass = ClassSampledValuesTrain.iloc[:, ClassColumnIndex].values.reshape(-1,1)# select class column only 

    testXClass = ClassSampledValuesTest.iloc[:,0:B]#select training data colums
    testyClass = ClassSampledValuesTest.iloc[:, ClassColumnIndex].values.reshape(-1,1)# select class column only 
  
    trainX =  np.append(trainX, trainXClass, axis = 0)
    testX = np.append(testX, testXClass, axis = 0)
    trainy = np.append(trainy,trainyClass, axis = 0)
    testy = np.append(testy, testyClass, axis = 0)

  #create machine learning model and determine accuracy
  trainy= trainy.astype(int)#  class format needs to be particular format - int works - for classifer
  testy= testy.astype(int)#  class format needs to be particular format - int works - for classifer

  rf = RandomForestClassifier( random_state = 99, max_features=2, n_estimators=1000)
  rf.fit(trainX, trainy) 
  pred_y = rf.predict(testX)
  cm = confusion_matrix(testy, pred_y)
  acc = accuracy_score(testy, pred_y)
  f1 = f1_score(testy, pred_y, average = None  )
  model = rf


  #write image name to file if accuracy (acc) <0.8, otherwise classify imagery as water/ not water and write output
  if acc <0.8:
    pass

   #write to file 
  else:
    #create new model based on combined training and validation data, apply to rest of scene.
    alltrainX = np.append(testX, trainX, axis = 0)
    alltrainy = np.append(testy, trainy, axis = 0)
    alltrainy = np.ndarray.flatten(alltrainy)
    rf.fit(alltrainX, alltrainy)

    #generate 2 extra bands for whole image
    #calculate indices and convert to float



   ##create 4 band image for AOI
   ##open AOI
    import fiona
    import rasterio.mask
    with fiona.open (os.path.join(AOI_dir,aoi_fn), "r")as shapefile:

      shape = [feature["geometry"] for feature in shapefile]
    with rasterio.open(image) as s1:
      out_image, out_transform = rasterio.mask.mask(s1,shape, crop=True)

      out_meta = s1.meta
      out_meta.update({"driver":"GTiff",
                              "height":out_image.shape[1],
                              "width":out_image.shape[2],
                              "transform":out_transform,
                              "count":4})
      s1prof.update({"count" : 1,
                     "nodata":None,
                     "dtype":np.float64, # Edit LH: array was float64 which caused mismatch with files float32
                     "height":out_image.shape[1],
                     "width":out_image.shape[2],
                     "transform":out_transform,
                     "count":1,
                     "driver":"GTiff"})
                     
      

      VVcrop = out_image[0,:,:].astype(float)
      VHcrop = out_image[1,:,:].astype(float)

    with rasterio.open(os.path.join(temp_dir,'_4_indices.tif'), 'w', **out_meta) as dst:
      
      VVVHratio = VVcrop/VHcrop
      VVVHdiff = VVcrop-VHcrop
      dst.write_band(1, VVcrop)
      dst.write_band(2, VHcrop)
      dst.write_band(3, VVVHratio)
      dst.write_band(4, VVVHdiff)
    dst.close()
      #Read whole image
    fourband = rasterio.open(os.path.join(temp_dir,'_4_indices.tif'))


    dst2 = rasterio.open(os.path.join(output_dir,'{}_water_land.tif'.format(imagename)), 'w', **s1prof)
    for block_index, window in fourband.block_windows(1):
        s1_block = fourband.read(window=window, masked=True)

        v= s1_block.shape
        s1_block = s1_block.reshape(B, -1).T

        s1_block[s1_block<-3.4e+35]=9999
        
        #deal with nan
        s1_block[np.isnan(s1_block)]=9999
        s1_block[np.isinf(s1_block)]=9999

        result_block = model.predict_proba(s1_block).astype('float64')

        #select probabilies for class 1 only
        #output binary classification - water 1/ not water 0
        result_block = result_block[:,0]
        result_block[result_block<0.7]=0 #0.7 threshold was used for Muckle water classification
        result_block[(result_block>=0.7)&(result_block<1.1)]=1
        result_block = result_block.reshape(1,v[1],v[2])
        dst2.write(result_block, window=window)
    fourband.close()
    dst2.close()


