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