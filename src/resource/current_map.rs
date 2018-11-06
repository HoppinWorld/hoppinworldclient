use hoppinworlddata::MapInfo;

/// Single row of the MapInfoCache
/// File name -> MapInfo
#[derive(Debug, Clone, new)]
pub struct CurrentMap(pub String, pub MapInfo);