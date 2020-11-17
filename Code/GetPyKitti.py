import pykitti




# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically. 
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.

def getPyKittiData(startFrames = 0, stopFrames=50, stepFrames=5):
    # basedir = r"../Datasets/KITTI_cvlibs/2011_09_26_drive_0001"
    basedir = r"/media/LinData/Datasets/KITTI_cvlibs/2011_09_26_drive_0001"
    date = '2011_09_26'
    drive = '0001'
    return pykitti.raw(basedir, date, drive, frames=range(startFrames, stopFrames, stepFrames))

""""

In case of error:
  - Check if the data is actually avalible at the correct base directory.
  - There should be a folder with the date in the directory.
  - In that folder (2011_09_26) there should be folders with the calibration, sync, tracklets and data for the drive recorded.
 
"""





if __name__ == "__main__":
    print(getPyKittiData())