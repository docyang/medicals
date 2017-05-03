import SimpleITK as sitk
import os

def read_images_annotations_from_medical_sources(path, num_threads=8):
    try:
        path.decode('ascii')
    except UnicodeDecodeError:
        ImageETL.Logger.debug("file path shall not contain unicode chars!")
        return None



    ns_image_list = []
    total_annotation_list = []
    series_reader = sitk.ImageSeriesReader()
    series_file_names = get_largest_dcm_series_recursively(path, series_reader)
    series_reader.SetOutputPixelType(sitk.sitkInt16)
    series_reader.SetNumberOfThreads(num_threads)

    for srf in series_file_names:
        file_reader = sitk.ImageFileReader()
        instance_index_map = {}
        position_index_map = {}
        for index, name in enumerate(srf):
            file_reader.SetOutputPixelType(sitk.sitkInt16)
            file_reader.SetNumberOfThreads(num_threads)
            file_reader.SetFileName(name)
            instance = file_reader.Execute()
            instance_number = int(trim(instance.GetMetaData(DCMTAGS.INSTANCE_NUMBER)))
            z_position = float(trim(instance.GetMetaData(DCMTAGS.SLICE_POSITION)))
            instance_index_map[instance_number] = (index, instance)
            position_index_map[z_position] = index
        first_instance = instance_index_map[min(instance_index_map.keys())][-1]
        metadata_dict = fetch_metadata_from_itkimage(first_instance)
        ns_image = NumpySImage()
        header = SImageHeader()
        header.application_specific_metadata = metadata_dict
        first_file_name = srf[0]
        header.application_specific_metadata[ORIGINAL_PATH] = first_file_name
        header.component_type = SImageHeader.COMPONENT_TYPE.GRAY
        header.data_type = SImageHeader.DATA_TYPE.SIGNED_SHORT
        series_reader.SetFileNames(srf)
        # series_reader.AddCommand(sitk.sitkProgressEvent, ReadProgressObserver(series_reader))
        series_image = series_reader.Execute()
        header.dim = series_image.GetDimension()
        header.size = list(series_image.GetSize())
        header.size[0], header.size[1] = header.size[1], header.size[0]
        header.origin = list(series_image.GetOrigin())
        header.origin[0], header.origin[1] = header.origin[1], header.origin[0]
        header.direction = series_image.GetDirection()
        header.direction = np.array(header.direction).reshape(header.dim, header.dim).tolist()
        header.direction[0], header.direction[1] = header.direction[1], header.direction[0]
        header.spacing = list(series_image.GetSpacing())
        header.spacing[0], header.spacing[1] = header.spacing[1], header.spacing[0]
        header.sample_per_component = series_image.GetNumberOfComponentsPerPixel()
        if header.sample_per_component != 1:
            ImageETL.Logger.debug("Series contains pixel with multiple channels!")
            return None
        image_array = sitk.GetArrayFromImage(series_image)
        image_array = convert_numpy_data_type(itk_zyx_to_yxz(image_array),
                                              NumpySImage.DTYPE_MAPPING[SImageHeader.DATA_TYPE.SIGNED_SHORT])
        ns_image.set_pixel_data(image_array)
        ns_image.set_header(header)
        gtd_path_list = AnnotationETL.fetch_gtd_names_from_image_dir(os.path.dirname(srf[0]))
        xml_path_list = AnnotationETL.fetch_xml_names_from_image_dir(os.path.dirname(srf[0]))
        annotation_list = []
        for gtd_file in gtd_path_list:
            single_annotation = AnnotationETL.gtd2annotation(gtd_file, instance_index_map)
            single_annotation.get_header().referenced_image_uid = header.image_uid
            annotation_list.append(single_annotation)

        for xml_file in xml_path_list:
            xml_annotations = AnnotationETL.lidc_xml2annotation(xml_file, position_index_map)
            for single_annotation in xml_annotations:
                single_annotation.get_header().referenced_image_uid = header.image_uid
                annotation_list.append(single_annotation)
        ns_image_list.append(ns_image)
        total_annotation_list.extend(annotation_list)
    return ns_image_list, total_annotation_list