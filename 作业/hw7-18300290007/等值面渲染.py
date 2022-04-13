import vtk
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

# 使用nibabel读入图像数据img_data中
img=nib.load('image_lr.nii.gz')
img_data = img.get_fdata()
dims = img.shape
spacing = (img.header['pixdim'][1], img.header['pixdim'][2], img.header['pixdim'][3])

# vtk的image data对象
image = vtk.vtkImageData()
image.SetDimensions(dims[0], dims[1], dims[2])
image.SetSpacing(spacing[0], spacing[1], spacing[2])
image.SetOrigin(0,0,0)

if vtk.VTK_MAJOR_VERSION<=5:
    image.SetNumberOfScalarComponents(1)
    image.SetScalarTypeToDouble()
else:
    image.AllocateScalars(vtk.VTK_DOUBLE,1)

# 逐点输入3d数据
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            scalardata = img_data[x][y][z]
            image.SetScalarComponentFromDouble(x,y,z,0,scalardata)

# 使用Marching Cude算法进行面渲染
Extractor = vtk.vtkMarchingCubes()
Extractor.SetInputData(image)

# 设置等值面
Extractor.SetValue(0, 100)
#Extractor.SetValue(1, 200)

# 先建立三角条带对象
stripper = vtk.vtkStripper()
stripper.SetInputConnection(Extractor.GetOutputPort()) #连接三角片
# 设置mapper，acter，renderer等
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1,1,0.5)
# actor.GetProperty().SetOpacity(0.9)
# actor.GetProperty().SetAmbient(0.25)
# actor.GetProperty().SetDiffuse(0.6)
# actor.GetProperty().SetSpecular(1.0)

# 生成交互式窗口
ren = vtk.vtkRenderer()
ren.SetBackground(1,1,1)
ren.AddActor(actor)
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(500, 500)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()
