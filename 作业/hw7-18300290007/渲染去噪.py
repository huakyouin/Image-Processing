import nibabel as nib
import vtk
import numpy as np
import tqdm 

#读取数据
img = nib.load("image_lr.nii.gz")
img_data = img.get_fdata()
dims = img.shape
spacing = (img.header['pixdim'][1], img.header['pixdim'][2], img.header['pixdim'][3])

#声明vtk的image类型
image = vtk.vtkImageData()
image.SetDimensions(dims[0], dims[1], dims[2])
image.SetSpacing(spacing[0], spacing[1], spacing[2])
image.SetOrigin(0,0,0)


if vtk.VTK_MAJOR_VERSION<=5:
    image.SetNumberOfScalarComponents(1)
    image.SetScalarTypeToDouble()
else:
    image.AllocateScalars(vtk.VTK_DOUBLE,1)


#------------------使用类似2d平滑滤波的方式进行3d平滑
#padding--零填充
m=np.zeros(shape=(dims[0]+2,dims[1]+2,dims[2]+2))
print('零填充进度')
for z in tqdm.tqdm(range(dims[2])):
    for y in range(dims[1]):
        for x in range(dims[0]):
            m[x+1][y+1][z+1]=img_data[x][y][z]
#平滑处理--均值法（3*3*3滑窗）
print('平滑处理进度')
for z in tqdm.tqdm(range(dims[2])):
    for y in range(dims[1]):
        for x in range(dims[0]):
            sumt=0
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    for k in [-1,0,1]:
                        sumt=sumt+m[x+k+1][y+j+1][z+i+1]
            sumt=sumt/27
            scalardata=int(sumt)
            image.SetScalarComponentFromDouble(x,y,z,0,scalardata)

# 沿用等值面渲染中方法
Extractor = vtk.vtkMarchingCubes()
Extractor.SetInputData(image)
Extractor.SetValue(0, 100)
#Extractor.SetValue(1, 200)

stripper = vtk.vtkStripper()
stripper.SetInputConnection(Extractor.GetOutputPort())
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1,1,0.5)

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
