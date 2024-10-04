# import vtk
#
# # VTP 파일 경로 설정
# file_path = "C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/Diseased/Centerlines/Diseased_1.vtp"
#
# # VTP 파일 읽기
# reader = vtk.vtkXMLPolyDataReader()
# reader.SetFileName(file_path)
# reader.Update()
#
# # 읽은 데이터 가져오기
# poly_data = reader.GetOutput()
#
# # 예시: 데이터 정보 출력
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputData(poly_data)
#
# # Actor 생성
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)
#
# # Renderer 생성
# renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
# renderer.SetBackground(0.1, 0.2, 0.4)  # 배경색 설정
#
# # Render Window 생성 및 설정
# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)
# render_window.SetSize(800, 600)
# render_window.SetWindowName("VTP Visualization")
#
# # Render Window Interactor 생성
# interactor = vtk.vtkRenderWindowInteractor()
# interactor.SetRenderWindow(render_window)
#
# # 시각화 시작
# interactor.Initialize()
# interactor.Start()
#
import vtk
import numpy as np
import matplotlib.pyplot as plt

vtp_file_path = "C:/Users/최재원/Desktop/ASOCADataAccess/ASOCADataAccess/CompressedVersion/Normal/Centerlines/Normal_4.vtp"
vtp_reader = vtk.vtkXMLPolyDataReader()
vtp_reader.SetFileName(vtp_file_path)
vtp_reader.Update()
vtp_poly_data = vtp_reader.GetOutput()

points = vtp_poly_data.GetPoints()

num_points = points.GetNumberOfPoints()
points_array = np.zeros((num_points, 3))
for i in range(num_points):
    points_array[i] = points.GetPoint(i)

x = points_array[:, 0]
y = points_array[:, 1]
z = points_array[:, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', marker='o', s=1)  # 점으로 표시

cells = vtp_poly_data.GetPolys()
cells.Reset()
for i in range(cells.GetNumberOfCells()):
    cell_points = vtk.vtkIdList()
    cells.GetNextCell(cell_points)
    if cell_points.GetNumberOfIds() == 3:  # 삼각형 셀만 고려
        polygon = np.array([[points.GetPoint(cell_points.GetId(0))],
                            [points.GetPoint(cell_points.GetId(1))],
                            [points.GetPoint(cell_points.GetId(2))]])
        polygon = np.vstack([polygon, polygon[0]])  # 폴리곤의 처음과 끝을 이어줌
        ax.plot(polygon[:, 0], polygon[:, 1], polygon[:, 2], c='black')  # 선으로 폴리곤 표시

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('VTP Data Visualization')
plt.show()