use MapInfoCache;

pub fn get_all_maps(base_path: &str) -> MapInfoCache {
    let maps_path = format!(
        "{}{}maps{}",
        base_path,
        std::path::MAIN_SEPARATOR,
        std::path::MAIN_SEPARATOR
    );

    let map_info_vec = std::fs::read_dir(&maps_path)
        .expect(&*format!("Failed to read maps directory {}.", &maps_path))
        .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_file())
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension()
                .unwrap_or_else(|| std::ffi::OsStr::new(""))
                .to_str()
                .unwrap()
                == "hop"
        })
        .map(|e| {
            let info_file_data = std::fs::read_to_string(e.to_str().unwrap()).unwrap();
            let info =
                ron::de::from_str(&info_file_data).expect("Failed to deserialize info map file.");

            Some((e.file_stem().unwrap().to_str().unwrap().to_string(), info))
        })
        .flatten()
        .collect::<Vec<_>>();
    MapInfoCache::new(map_info_vec)
}

pub fn gltf_path_from_map(base_path: &str, map_name: &str) -> String {
    format!(
        "{}{}maps{}{}.glb",
        base_path,
        std::path::MAIN_SEPARATOR,
        std::path::MAIN_SEPARATOR,
        map_name
    )
}
