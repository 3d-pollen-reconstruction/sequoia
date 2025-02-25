import json
import os
import random

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def render_model_views_from_file(selected_path, rotation):
    # Read the STL file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(selected_path)
    reader.Update()

    # Apply smoothing filter to simplify the mesh
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputConnection(reader.GetOutputPort())
    smoothFilter.SetNumberOfIterations(30)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOff()
    smoothFilter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(smoothFilter.GetOutputPort())

    # Create two actors sharing the same mapper (for two views)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper)

    # Set material and shading properties for a reflective, grayish look.
    for actor in (actor1, actor2):
        actor.GetProperty().SetColor(1, 1, 1)  # white/gray
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(50)

    # Create two renderers for the two viewports (left and right)
    renderer1 = vtk.vtkRenderer()
    renderer2 = vtk.vtkRenderer()

    renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
    renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

    renderer1.SetBackground(0, 0, 0)
    renderer2.SetBackground(0, 0, 0)

    renderer1.AddActor(actor1)
    renderer2.AddActor(actor2)

    # Set up lighting
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(1, 1, 1)
    light.SetFocalPoint(0, 0, 0)
    light.SetColor(1, 1, 1)
    light.SetIntensity(1.0)
    renderer1.AddLight(light)
    renderer2.AddLight(light)

    # Optional shadow mapping.
    shadow_pass = vtk.vtkShadowMapPass()
    opaque_pass = vtk.vtkOpaquePass()
    render_pass_collection = vtk.vtkRenderPassCollection()
    render_pass_collection.AddItem(opaque_pass)
    render_pass_collection.AddItem(shadow_pass)
    sequence_pass = vtk.vtkSequencePass()
    sequence_pass.SetPasses(render_pass_collection)
    camera_pass = vtk.vtkCameraPass()
    camera_pass.SetDelegatePass(sequence_pass)
    renderer1.SetPass(camera_pass)
    renderer2.SetPass(camera_pass)

    # Centering the object in view
    polydata = smoothFilter.GetOutput()
    bounds = polydata.GetBounds()
    center = polydata.GetCenter()
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    distance = max_dim * 2.5 if max_dim > 0 else 100

    # Camera for view 1 (from the Z axis)
    camera1 = vtk.vtkCamera()
    camera1.SetFocalPoint(center)
    camera1.SetPosition(center[0], center[1], center[2] + distance)
    camera1.SetViewUp(0, 1, 0)
    renderer1.SetActiveCamera(camera1)

    # Camera for view 2 (from the X axis)
    camera2 = vtk.vtkCamera()
    camera2.SetFocalPoint(center)
    camera2.SetPosition(center[0] + distance, center[1], center[2])
    camera2.SetViewUp(0, 1, 0)
    renderer2.SetActiveCamera(camera2)

    # Apply the provided rotation to both actors
    actor1.SetOrigin(center)
    actor2.SetOrigin(center)
    actor1.SetOrientation(rotation)
    actor2.SetOrientation(rotation)

    # Setup offscreen render window.
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(2048, 1024)  # 1024x1024 for each view side-by-side
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer1)
    renderWindow.AddRenderer(renderer2)
    renderer1.ResetCameraClippingRange()
    renderer2.ResetCameraClippingRange()
    renderWindow.Render()

    # Capture screenshot of the combined view.
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renderWindow)
    w2if.Update()

    vtk_image = w2if.GetOutput()
    dims = vtk_image.GetDimensions()  # (width, height, depth)
    num_comp = vtk_image.GetPointData().GetScalars().GetNumberOfComponents()
    vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    height, width = dims[1], dims[0]
    arr = vtk_array.reshape(height, width, num_comp)
    arr = np.flipud(arr)  # Flip vertically for correct orientation

    # Convert RGB image to grayscale using standard luminance conversion.
    if num_comp >= 3:
        arr = np.dot(arr[...,:3], [0.299, 0.587, 0.114])
    else:
        arr = arr[..., 0]

    # Split the grayscale image into left and right views.
    half_width = width // 2
    left_view = arr[:, :half_width]
    right_view = arr[:, half_width:]

    return left_view, right_view


def render_random_pollen():
    """
    Selects a random STL file (from 'data/models' excluding certain files),
    generates a random rotation, and then renders the object.
    Returns:
        selected_path (str): The chosen model file path.
        random_rotation (tuple): The (x, y, z) rotation applied.
        left_view, right_view (np.ndarray): The two rendered views.
    """
    model_dir = "data/models"
    pollen_to_exclude = [
        # Big Holes:
        "17778_Salix%20alba%20-%20Willow_NIH3D.stl",
        "17796_Fuchsia%20magellanica%20-%20Hardy%20fuchsia_NIH3D.stl",
        "17806_anthriscus_sylvestris-cow_parsley_NIH3D.stl",
        "17829_Rhododendron-Rhododendron_NIH3D.stl",
        "17831_Ilex%20aquifolium%20-%20Common%20holly_NIH3D.stl",
        "17843_Potamogeton%20lucens%20-%20Pondweed_NIH3D.stl",
        "17844_Pinus%20sylvestris%20-%20Scots%20pine_NIH3D.stl",
        "17845_narcissus_sp-daffodil_NIH3D.stl",
        "17846_Polypodium%20vulgare%20-%20Common%20fern_NIH3D.stl",
        "17878_alnus_sp-alder-pentaporate_NIH3D.stl",
        "20074_typha_latifolia-bulrush-linear_NIH3D.stl",
        "20076_thlaspi_arvense_2_NIH3D.stl",
        "20605_acer_campestre_NIH3D.stl",
        "20609_artemisia_absinthium_NIH3D.stl",
        "20702_pedicularis_sylvatica_a_NIH3D.stl",
        "20706_potentilla_reptans_NIH3D.stl",
        "20854_castanea_sativa_NIH3D.stl",
        "20855_galium_verum_1_NIH3D.stl",
        "20861_origanum_vulgare_NIH3D.stl",
        "20862_polygala_serpyllifolia_a_NIH3D.stl",
        "20939_senecio_vulgaris_NIH3D.stl",
        "20944_viola_tricolor_NIH3D.stl",
        "21089_bellis_perennis_NIH3D.stl",
        "21105_picris_hieracioides_NIH3D.stl",
        "21106_petasites_hybridus_NIH3D.stl",
        "21112_lysimachia_nemorum_NIH3D.stl",
        "21185_tamarix_anglica_NIH3D.stl",
        "21186_thesium_humifusum_NIH3D.stl",
        "21191_umbilicus_rupestris_NIH3D.stl",
        "21253_salix_phylicifolia_NIH3D.stl",
        "21256_scleranthus_annuus_2_NIH3D.stl",
        "21257_umbilicus_rupestris_NIH3D.stl",
        "21264_solidago_virgaurea_NIH3D.stl",
        "21270_sedum_acre_NIH3D.stl",
        "21279_sagittaria_sagittifolia_NIH3D.stl",
        "21280_saxifraga_aizoides_NIH3D.stl",
        "21285_saxifraga_stellaris_NIH3D.stl",
        "21465_saxifraga_stellaris_NIH3D.stl",
        "21534_tussilago_farfara_NIH3D.stl",
        "21536_valerianella_locusta_NIH3D.stl",
        "21538_viburnum_lantana_NIH3D.stl",
        "21541_jacobaea_vulgaris_NIH3D.stl",
        "21542_vaccinium_myrtillus_NIH3D.stl",
        "21552_verbascum_nigrum_NIH3D.stl",
        "21553_veronica_anagallis-aquatica_NIH3D.stl",
        "21554_veronica_arvensis_NIH3D.stl",
        "21557_viola_hirta_NIH3D.stl",
        "21583_valerianella_locusta_NIH3D.stl",
        "21602_pulicaria_dysenterica_NIH3D.stl",
        "21628_prunus_padus_NIH3D.stl",
        # Lots of holes:
        "17811_Chenopodium%20album%20-%20Goosefoot_NIH3D.stl",
        "17884_Spathiphyllum%20cannifolium_NIH3D.stl",
        "20470_buxus_sempervirens_NIH3D.stl",
        "20471_atriplex_hastata_NIH3D.stl",
        "20610_atriplex_hastata_NIH3D.stl",
        "20608_arenaria_serpyllifolia_NIH3D.stl",
        "21184_dioscorea_communis_NIH3D.stl",
        "21190_triglochin_maritima_NIH3D.stl",
        "21271_silene_alba_NIH3D.stl",
        "21584_luzula_sylvatica_NIH3D.stl",
        "21628_primula_veris_NIH3D.stl",
        # Has a hair or something:
        "17883_Oenothera%20fruticosa_NIH3D.stl",
        # Crazy shape:
        "17900_Germinating%20lily%20pollen_NIH3D.stl",
        "21252_Germinating%20lily%20pollen_NIH3D.stl",
        # Probably burst open:
        "20707_primula_vulgaris_NIH3D.stl",
        "20943_veronica_agrestis_NIH3D.stl",
        "21266_sherardia_arvensis_NIH3D.stl",
        # Probably heavily collapsed:
        "20468_juniperis_communis_NIH3D.stl",
        "20703_larix_decidua_1_NIH3D.stl",
        "20858_larix_decidua_1_NIH3D.stl",
        "20859_larix_decidua_2_NIH3D.stl",
        "20932_bromus_ramosus_NIH3D.stl",
        "20934_taxus_baccata_NIH3D.stl",
        "20936_pteridium_aquilinum_NIH3D.stl",
        "21268_secale_cereale_2_NIH3D.stl",
        "21375_phleum_pratense_NIH3D.stl",
        "21376_festuca_rubra_2_NIH3D.stl",
        "21464_anthoxanthum_odoratum_NIH3D.stl",
        "21467_anthoxanthum_odoratum_NIH3D.stl",
        "21469_phegopteris_connectilis_NIH3D.stl",
        "21470_phegopteris_connectilis_2_NIH3D.stl",
        "21531_bromus_hordeaceus_NIH3D.stl",
        "21551_arrhenatherum_elatius_NIH3D.stl",
        "21600_polytrichum_commune_NIH3D.stl",
    ]
    all_files = os.listdir(model_dir)
    filtered_files = [
        f for f in all_files if f.endswith(".stl") and f not in pollen_to_exclude
    ]

    if not filtered_files:
        raise RuntimeError("No STL files left after filtering.")

    # Choose a random file and generate a random rotation.
    selected_file = random.choice(filtered_files)
    selected_path = os.path.join(model_dir, selected_file)
    random_rotation = (
        random.uniform(0, 360),
        random.uniform(0, 360),
        random.uniform(0, 360),
    )
    left_view, right_view = render_model_views_from_file(selected_path, random_rotation)
    return selected_path, random_rotation, left_view, right_view


if __name__ == "__main__":
    # Render a random pollen and get the view images.
    selected_path, random_rotation, left_view, right_view = render_random_pollen()

    print("Left view dimensions:", left_view.shape)
    print("Right view dimensions:", right_view.shape)

    # Save the screenshot (concatenating both views).
    combined_view = np.concatenate((left_view, right_view), axis=1)
    from PIL import Image

    # Convert grayscale image back to RGB by replicating the single channel
    combined_view_rgb = np.stack((combined_view,) * 3, axis=-1)
    image = Image.fromarray(combined_view_rgb.astype(np.uint8))
    image.save("combined_views.png")

    # Save the random rotation to a JSON file.
    with open("object_rotation.json", "w") as f:
        json.dump({"rotation": random_rotation}, f)

    print("Selected file:", selected_path)
    print("Stored random rotation augmentation:", random_rotation)
    print("Saved screenshot to combined_views.png")
