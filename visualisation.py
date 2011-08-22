# -*- coding: utf-8 -*-  

from vtk import *

#--- visualisation ---------------------------------------->>>
def visualisate(OutputPort):
  """
  VTK-s megjelenítés
  """
  planeSource = vtkPlaneSource();
  planeSource.Update();

  mapper = vtkPolyDataMapper();
  mapper.SetInput(planeSource.GetOutput());

  texture = vtkTexture();
  #texture.SetInputConnection(reader.GetOutputPort());
  texture.SetInputConnection(OutputPort);

  actor = vtkActor();
  actor.SetTexture(texture);
  actor.SetMapper(mapper);

  renderer = vtkRenderer();
  renderer.AddActor(actor);
  renderer.SetBackground(0.5,0.7,0.7);
  
  renWin = vtkRenderWindow();
  renWin.AddRenderer(renderer);

  interactor = vtkRenderWindowInteractor();
  interactor.SetRenderWindow(renWin);

  renWin.SetSize(650,650);
  renWin.Render();
  interactor.Initialize();
  interactor.Start();
#--- end block: "visualisation" ---------------------------<<<