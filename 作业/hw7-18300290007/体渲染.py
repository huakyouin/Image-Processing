import nibabel as nib
import vtk
import numpy as np

#数据读取和定义
img = nib.load("image_lr.nii.gz")
img_data = img.get_fdata()
dims = img.shape
spacing = (img.header['pixdim'][1], img.header['pixdim'][2], img.header['pixdim'][3])

#vtk的image对象声明
image = vtk.vtkImageData()
image.SetDimensions(dims[0], dims[1], dims[2])
image.SetSpacing(spacing[0], spacing[1], spacing[2])
image.SetOrigin(0,0,0)

#逐点赋值
image.AllocateScalars(vtk.VTK_DOUBLE, 1)
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            scalardata = img_data[x][y][z]
            image.SetScalarComponentFromDouble(x,y,z,0,scalardata) 

volumeProperty = vtk.vtkVolumeProperty()

#-------------设置传输函数参数，进行体渲染
#创建将标量值转换为不透明度的映射
compositeOpacity = vtk.vtkPiecewiseFunction()
compositeOpacity.AddSegment(0, 0, 10, 0)
compositeOpacity.AddSegment(10, 0.2, 120, 0.2)
#compositeOpacity.AddSegment(120, 0.2, 128, 0.4)
volumeProperty.SetScalarOpacity(compositeOpacity)

# 上色
colorFunction = vtk.vtkColorTransferFunction()
colorFunction.AddRGBSegment(0, 0, 0, 0, 20, 0.2, 0.2, 0.2)
#colorFunction.AddRGBSegment(10, 0.94, 0.9, 0.55, 120, 0.94, 0.9, 0.55)
colorFunction.AddRGBSegment(20, 0.1, 0.1, 0, 128, 1, 1, 0)
volumeProperty.SetColor(colorFunction)

# 标量值+梯度模
gradientTransferFunction=vtk.vtkPiecewiseFunction()
gradientTransferFunction.AddPoint(0,0.0)
gradientTransferFunction.AddSegment(100, 0.1, 1000, 0.3)
volumeProperty.SetGradientOpacity(gradientTransferFunction)

# 光学模拟
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetAmbient(1) 
volumeProperty.SetDiffuse(0.9) # 漫反射
volumeProperty.SetSpecular(0.5) # 镜面反射
volumeProperty.SetSpecularPower(10)

# 光线投射量绘制
volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
volumeMapper.SetInputData(image)
volumeMapper.SetImageSampleDistance(5.0)
# volume包含映射器和属性以及可用于定位体积
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

#------------结果生成
ren = vtk.vtkRenderer()
ren.SetBackground(1,1,1)
ren.AddActor(volume)
light=vtk.vtkLight()
light.SetColor(0,1,1)
ren.AddLight(light)
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(500, 500)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()