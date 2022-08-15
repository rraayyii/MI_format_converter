import vtk
import SimpleITK as sitk
import numpy as np
from stl import mesh
import os
from os import path

def nii2stl(filename_nii, filename):
    # can be done in a loop if you have multiple files to be processed, speed is guaranteed if GPU is used:)

    # read all the labels present in the file
    multi_label_image = sitk.ReadImage(filename_nii)
    img_npy = sitk.GetArrayFromImage(multi_label_image)
    labels = np.unique(img_npy)

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()

    # for all labels presented in the segmented file
    for label in labels:

        if int(label) != 0:

            # apply marching cube surface generation
            surf = vtk.vtkDiscreteMarchingCubes()
            surf.SetInputConnection(reader.GetOutputPort())
            surf.SetValue(0,
                          int(label))  # use surf.GenerateValues function if more than one contour is available in the file
            surf.Update()

            # smoothing the mesh
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            if vtk.VTK_MAJOR_VERSION <= 5:
                smoother.SetInput(surf.GetOutput())
            else:
                smoother.SetInputConnection(surf.GetOutputPort())

            # increase this integer set number of iterations if smoother surface wanted
            smoother.SetNumberOfIterations(30)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
            smoother.GenerateErrorScalarsOn()
            smoother.Update()

            """added"""
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputConnection(smoother.GetOutputPort())
            decimate.SetTargetReduction(.95)
            decimate.Update()


            # save the output
            writer = vtk.vtkSTLWriter()
            writer.SetInputConnection(decimate.GetOutputPort())
            writer.SetFileTypeToASCII()

            # file name need to be changed
            # save as the .stl file, can be changed to other surface mesh file
            # writer.SetFileName(f'{filename}_{label}.stl')
            writer.SetFileName(filename)
            writer.Write()

            #recenter the mesh
            segMesh = mesh.Mesh.from_file(filename)
            offset = multi_label_image.GetOrigin()
            segMesh.x += offset[0]
            segMesh.y += offset[1]
            segMesh.z += offset[2]
            newUSMeshFN = filename
            segMesh.save(newUSMeshFN)

if __name__ == '__main__':
    # source = 'D:/uronav_data/Case0001/ps_mr_label.nii.gz'
    # target = 'D:/uronav_data/Case0001/ps_mr_label.stl'
    # nii2stl(source, target)
    data_dir = 'D:/uronav_data'
    for casename in os.listdir(data_dir):
        if 'Case' in casename:
            lb_fn = path.join(data_dir, casename, 'ps_us_label.nii.gz')
            stl_fn = path.join(data_dir, casename, 'us_ps.stl')
            nii2stl(lb_fn, stl_fn)
            print(casename)
