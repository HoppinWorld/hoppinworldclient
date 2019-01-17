use hoppinworlddata::MapInfo;

#[derive(Default, Debug, Clone, Serialize, Deserialize, new)]
pub struct MapInfoCache {
    /// File name -> MapInfo
    pub maps: Vec<(String, MapInfo)>,
}
