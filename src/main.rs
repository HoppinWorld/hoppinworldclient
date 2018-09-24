extern crate amethyst;
extern crate amethyst_extra;
extern crate amethyst_gltf;
extern crate amethyst_rhusics;
#[macro_use]
extern crate serde;
#[macro_use]
extern crate log;
extern crate partial_function;
extern crate winit;
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate specs_derive;
extern crate ron;
extern crate uuid;
extern crate hoppinworlddata;
#[macro_use]
extern crate derive_builder;

use std::collections::VecDeque;
use hoppinworlddata::*;
use amethyst::assets::Prefab;
use amethyst::assets::RonFormat;
use amethyst::assets::{
    AssetPrefab, PrefabData, PrefabLoader, PrefabLoaderSystem, ProgressCounter,
};
use amethyst::controls::{
    CursorHideSystem, FlyControlTag, HideCursor, MouseFocusUpdateSystem, WindowFocus,
};
use amethyst::core::transform::{Parent, Transform, TransformBundle};
use amethyst::core::{Named, Time};
use amethyst::ecs::error::Error as ECSError;
use amethyst::ecs::prelude::ParallelIterator;
use amethyst::ecs::{
    Component, DenseVecStorage, Entities, Entity, Join, ParJoin, Read, ReadStorage, Resources,
    System, SystemData, Write, WriteStorage,
};
use amethyst::input::{is_key_down, InputBundle, InputHandler};
use amethyst::prelude::*;
use amethyst::renderer::*;
use amethyst::shrev::{EventChannel, ReaderId};
use amethyst::ui::*;
use amethyst::utils::scene::BasicScenePrefab;
use amethyst_gltf::*;
use std::hash::Hash;

use amethyst::core::cgmath::{
    Basis3, Deg, EuclideanSpace, InnerSpace, Matrix3, One, Point3, Quaternion, Rotation3,
    SquareMatrix, Vector2, Vector3,
};
use amethyst_extra::*;
use amethyst_rhusics::collision::dbvt::query_ray;
use amethyst_rhusics::collision::primitive::{ConvexPolyhedron, Cylinder, Primitive3};
use amethyst_rhusics::collision::{Aabb3, Ray3};
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    Collider, CollisionMode, CollisionShape, CollisionStrategy, ContactEvent, ForceAccumulator,
    Material, NextFrame, PhysicalEntity, Pose, WorldParameters,
};
use amethyst_rhusics::rhusics_ecs::physics3d::{BodyPose3, DynamicBoundingVolumeTree3};
use amethyst_rhusics::rhusics_ecs::{PhysicalEntityParts, WithPhysics};
use amethyst_rhusics::{time_sync, DefaultPhysicsBundle3};
use partial_function::*;
use std::marker::PhantomData;
use uuid::Uuid;
use winit::DeviceEvent;

type ScenePrefab = BasicScenePrefab<Vec<PosNormTex>>;
type Shape = CollisionShape<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>;
type DefaultPhysicalEntityParts<'a, T> = PhysicalEntityParts<
    'a,
    Primitive3<f32>,
    T,
    Quaternion<f32>,
    Vector3<f32>,
    Vector3<f32>,
    Matrix3<f32>,
    Aabb3<f32>,
    BodyPose3<f32>,
>;
type MyPhysicalEntityParts<'a> = DefaultPhysicalEntityParts<'a, ObjectType>;
type CustomTrans<'a, 'b> = Trans<GameData<'a, 'b>, CustomStateEvent>;

const DISPLAY_SPEED_MULTIPLIER: f32 = 50.0;

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize, Copy)]
pub enum CustomStateEvent {
    // Actually a redirect to MapSelectState in this case.
    GotoMainMenu,
    MapFinished,
}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub enum RemovalId {
    Scene,
    GameplayUi,
    MenuUi,
    PauseUi,
    ResultUi,
    MapSelectUi,
}

#[repr(u8)]
#[derive(Debug, Clone, PartialOrd, PartialEq, Component)]
pub enum ObjectType {
    Scene,
    StartZone,
    EndZone,
    KillZone,
    Player,
    Dynamic,
    SegmentZone(u8),
}

impl Default for ObjectType {
    fn default() -> Self {
        ObjectType::Scene
    }
}

impl Collider for ObjectType {
    fn should_generate_contacts(&self, other: &ObjectType) -> bool {
        *self == ObjectType::Player || *other == ObjectType::Player || *self == ObjectType::Dynamic || *other == ObjectType::Dynamic
    }
}



#[derive(Serialize, Deserialize, Debug, Clone, new)]
struct Gravity {
    pub acceleration: Vector3<f32>,
}

impl Default for Gravity {
    fn default() -> Self {
        Gravity {
            acceleration: Vector3::new(0.0, -1.0, 0.0),
        }
    }
}

struct GravitySystem;

impl<'a> System<'a> for GravitySystem {
    type SystemData = (
        Read<'a, Time>,
        Read<'a, Gravity>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );
    fn run(&mut self, (time, gravity, mut velocities): Self::SystemData) {
        for (mut velocity,) in (&mut velocities,).join() {
            // Add the acceleration to the velocity.
            let new_vel = velocity.value.linear() + gravity.acceleration * time.delta_seconds();
            velocity.value.set_linear(new_vel);
        }
    }
}

#[derive(Debug, new, Clone, Serialize, Deserialize)]
pub struct PlayerSettings {
    pub grounded: Grounded,
    pub movement: BhopMovement3D,
    pub ground_friction: GroundFriction3D,
    pub shape: Primitive3<f32>,
    pub physical_entity: PhysicalEntity<f32>,
    pub mass: f32,
    pub gravity: f32,
    pub jump_velocity: f32,
}

#[derive(Deserialize, Serialize)]
pub struct PlayerPrefabData {
    grounded: Grounded,
    movement: BhopMovement3D,
    ground_friction: GroundFriction3D,
    shape: Primitive3<f32>,
    physical_entity: PhysicalEntity<f32>,
    mass: Mass3<f32>,
}

/*impl<'a> PrefabData<'a> for PlayerPrefabData {
    type SystemData = (
        <WriteStorage<'a, Grounded> as PrefabData<'a>>::SystemData,
        <WriteStorage<'a, BhopMovement3D> as PrefabData<'a>>::SystemData,
        <WriteStorage<'a, GroundFriction3D> as PrefabData<'a>>::SystemData,
        <Write<'a, PlayerSettings> as PrefabData<'a>>::SystemData,
    );
    type Result = ();

    fn load_prefab(
        &self,
        entity: Entity,
        system_data: &mut Self::SystemData,
        _entities: &[Entity],
    ) -> Result<(), ECSError> {
        let (ref mut groundeds, ref mut movements, ref mut frictions, ref mut player_settings) = system_data;
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");
        println!("STUFF GOING ON!");

        groundeds.insert(entity, self.grounded.clone())?;
        movements.insert(entity, self.movement.clone())?;
        frictions.insert(entity, self.ground_friction.clone())?;
        player_settings.shape = self.shape.clone();
        player_settings.physical_entity = self.physical_entity.clone();
        player_settings.mass = self.mass.clone();
        Ok(())
    }
}*/


/// Calculates in relative time using the internal engine clock.
#[derive(Default)]
pub struct RelativeTimer {
    pub start: f64,
    pub current: f64,
    pub running: bool,
}

impl RelativeTimer {
    pub fn get_text(&self) -> String {
        ((self.duration() * 1000.0).ceil() / 1000.0).to_string()
    }
    pub fn duration(&self) -> f64 {
        self.current - self.start
    }
    pub fn start(&mut self, cur_time: f64) {
        self.start = cur_time;
        self.current = cur_time;
        self.running = true;
    }
    pub fn update(&mut self, cur_time: f64) {
        if self.running {
            self.current = cur_time;
        }
    }
    pub fn stop(&mut self) {
        self.running = false;
    }
}

pub struct RuntimeProgress {
    pub current_segment: u8,
    pub segment_count: u8,
    pub segment_times: Vec<f32>,
}

impl Default for RuntimeProgress {
    fn default() -> Self {
        RuntimeProgress {
            current_segment: 1u8,
            segment_count: 0u8,
            segment_times: vec![],
        }
    }
}

impl RuntimeProgress {
    pub fn new(segment_count: u8) -> Self {
        RuntimeProgress {
            current_segment: 1u8,
            segment_count,
            // +1 to take into account last segment to end zone
            segment_times: vec![0.0; (segment_count + 1) as usize],
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeMap {
	pub start_zone: Entity,
	pub end_zone: Entity,
	pub segment_zones: Vec<(u8, Entity)>,
}

#[derive(Default)]
pub struct RuntimeMapBuilder {
	pub start_zone: Option<Entity>,
	pub end_zone: Option<Entity>,
	pub segment_zones: Vec<(u8, Entity)>,
}

impl RuntimeMapBuilder {
	pub fn start_zone(mut self, entity: Entity) -> Self {
		self.start_zone = Some(entity);
		self
	}

	pub fn end_zone(mut self, entity: Entity) -> Self {
		self.end_zone = Some(entity);
		self
	}

	pub fn build(mut self, map_info: &MapInfo) -> Result<RuntimeMap, String> {
		let start_zone = self.start_zone.ok_or("StartZone is not present in map.")?;
		let end_zone = self.end_zone.ok_or("EndZone is not present in map.")?;
        
        order_segment_zones(&mut self.segment_zones);
        validate_segment_zones(&self.segment_zones, map_info)?;
        Ok(RuntimeMap {
        	start_zone,
        	end_zone,
        	segment_zones: self.segment_zones,
        })
    }
}

pub fn order_segment_zones(segment_zones: &mut Vec<(u8, Entity)>) {
	segment_zones.sort_unstable_by(|a,b| a.0.cmp(&b.0));
}

/// Checks if the number of segments matches the number of segments indicated in the map info file.
/// Checks if the segment zones are in order and continuous and start at 1 and don't go over 254 and are not duplicated.
/// Expects the segment zones to be ordered by id.
pub fn validate_segment_zones(segment_zones: &Vec<(u8, Entity)>, map_info: &MapInfo) -> Result<(), String> {
	if segment_zones.len() > 254 {
		return Err(format!("Failed to load map: Too many segment zones (max 254)."));
	}

	if (segment_zones.len() + 1) as u8 != map_info.segment_count {
		return Err(format!("Failed to load map:\nThe segment zones count in the gltf (glb) map file + 1
			doesn't match the segment_count value of the map info file (.hop)\n
			glTF: {} + 1, map info: {}",
			segment_zones.len(),
			map_info.segment_count
		));
	}

	let mut last = 0u8;
	for seg_id in segment_zones.iter().map(|t| t.0) {
		// Duplicate id.
		if seg_id == last {
			return Err(format!("Failed to load map: Two segment zones have the same id: {}", seg_id));
		}
		// Non-continuous id distribution.
		if seg_id != last + 1u8 {
			return Err(format!("Failed to load map: There is a gap in the segment zone id. Jumped from {} to {}", last, seg_id));
		}
		last = seg_id;
	}

	// Good to go!
	Ok(())
}


pub struct RelativeTimerSystem;

impl<'a> System<'a> for RelativeTimerSystem {
    type SystemData = (Write<'a, RelativeTimer>, Read<'a, Time>);
    fn run(&mut self, (mut timer, time): Self::SystemData) {
        timer.update(time.absolute_time_seconds());
    }
}

#[derive(Default, Component, Serialize, Deserialize)]
pub struct Player;

/// Very game dependent.
/// Don't try to make that generic.
#[derive(Default)]
pub struct ContactSystem {
    contact_reader: Option<ReaderId<ContactEvent<Entity, Point3<f32>>>>,
}

impl<'a> System<'a> for ContactSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Transform>,
        Read<'a, EventChannel<ContactEvent<Entity, Point3<f32>>>>,
        Write<'a, RelativeTimer>,
        Read<'a, Time>,
        ReadStorage<'a, ObjectType>,
        ReadStorage<'a, Player>,
        ReadStorage<'a, BhopMovement3D>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
        WriteStorage<'a, NextFrame<BodyPose3<f32>>>,
        Write<'a, EventChannel<CustomStateEvent>>,
        Write<'a, RuntimeProgress>,
    );

    fn run(
        &mut self,
        (
            entities,
            transforms,
            contacts,
            mut timer,
            time,
            object_types,
            players,
            bhop_movements,
            mut velocities,
            mut body_poses,
            mut state_eventchannel,
            mut runtime_progress,
        ): Self::SystemData,
    ) {
        for contact in contacts.read(&mut self.contact_reader.as_mut().unwrap()) {
            //info!("Collision: {:?}",contact);
            let type1 = object_types.get(contact.bodies.0);
            let type2 = object_types.get(contact.bodies.1);

            if type1.is_none() || type2.is_none() {
                continue;
            }
            let type1 = type1.unwrap();
            let type2 = type2.unwrap();

            let (_player, other, player_entity) = if *type1 == ObjectType::Player {
                //(contact.bodies.0,contact.bodies.1)
                (type1, type2, contact.bodies.0)
            } else if *type2 == ObjectType::Player {
                //(contact.bodies.1,contact.bodies.0)
                (type2, type1, contact.bodies.1)
            } else {
                continue;
            };

            match other {
                ObjectType::StartZone => {
                    timer.start(time.absolute_time_seconds());
                    // Also limit player velocity while touching the StartZone to prevent any early starts.
                    // Not sure if this should go into state or not. Since it is heavily related to gameplay I'll put it here.
                    for (entity, _, movement, mut velocity) in
                        (&*entities, &players, &bhop_movements, &mut velocities).join()
                    {
                        if entity == player_entity {
                            let max_vel = movement.max_velocity_ground;
                            let cur_vel3 = *velocity.value.linear();
                            let mut cur_vel_flat = Vector2::new(cur_vel3.x, cur_vel3.z);
                            let cur_vel_flat_mag = cur_vel_flat.magnitude();
                            if cur_vel_flat_mag >= max_vel {
                                cur_vel_flat = cur_vel_flat.normalize() * max_vel;
                                velocity.value.set_linear(Vector3::new(
                                    cur_vel_flat.x,
                                    cur_vel3.y,
                                    cur_vel_flat.y,
                                ))
                            }
                        }
                    }

                    info!("start zone!");
                }
                ObjectType::EndZone => {
                    timer.stop();
                    info!("Finished! time: {:?}", timer.duration());
                    let id = runtime_progress.segment_count as usize;
                    runtime_progress.segment_times[id] = timer.duration() as f32;
                    state_eventchannel.single_write(CustomStateEvent::MapFinished);
                }
                ObjectType::KillZone => {
                    info!("you are ded!");
                    let seg = runtime_progress.current_segment;
                    let pos = if seg == 1 {
                        // To start zone
                        (&transforms, &object_types).join().filter(|(_,obj)| **obj == ObjectType::StartZone).map(|(tr,_)| tr.translation).next().unwrap()
                    } else {
                        // To last checkpoint

                        // Find checkpoint corresponding to the current segment in progress
                        (&transforms, &object_types).join().filter(|(_,obj)| {
                        	if let ObjectType::SegmentZone(s) = **obj {
                        		s == seg - 1
                        	} else {
                        		false
                        	}
                        }).map(|(tr,_)| tr.translation).next().unwrap()
                    };

                    // Move the player
                    let mut body_pose = (&players, &mut body_poses).join().map(|t| t.1).next().unwrap();
                    let pos = Point3::new(pos.x, pos.y, pos.z);
                    body_pose.value.set_position(pos);
                }
                ObjectType::SegmentZone(id) => {
                    if *id + 1 > runtime_progress.current_segment {
                        runtime_progress.segment_times[(*id) as usize] = timer.duration() as f32;
                        runtime_progress.current_segment = *id + 1;
                    }
                    info!("segment done");
                }
                _ => {}
            }
        }
    }

    fn setup(&mut self, res: &mut Resources) {
        Self::SystemData::setup(res);
        self.contact_reader = Some(
            res.fetch_mut::<EventChannel<ContactEvent<Entity, Point3<f32>>>>()
                .register_reader(),
        );
    }
}

pub struct RuntimeStats {
	pub jumps: u32,
	pub strafes: u32,
	pub jump_timings: VecDeque<(f64, f32)>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, new)]
pub struct MapInfoCache {
    /// File name -> MapInfo
    pub maps: Vec<(String, MapInfo)>,
}

/// Single row of the MapInfoCache
/// File name -> MapInfo
#[derive(Debug, Clone, new)]
pub struct CurrentMap(String, MapInfo);

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
        }).map(|e| {
            let info_file_data = std::fs::read_to_string(e.to_str().unwrap()).unwrap();
            let info =
                ron::de::from_str(&info_file_data).expect("Failed to deserialize info map file.");

            Some((e.file_stem().unwrap().to_str().unwrap().to_string(), info))
        }).flatten()
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

/// Very game dependent.
pub struct UiUpdaterSystem;

impl<'a> System<'a> for UiUpdaterSystem {
    type SystemData = (
        Read<'a, RelativeTimer>,
        Read<'a, PlayerStats>,
        ReadStorage<'a, Velocity3<f32>>,
        ReadStorage<'a, Jump>,
        ReadStorage<'a, UiTransform>,
        WriteStorage<'a, UiText>,
        ReadStorage<'a, Player>,
        Read<'a, RuntimeProgress>,
    );

fn run(&mut self, (timer, _stat, velocities, _jumps, ui_transforms, mut texts, players, runtime_progress): Self::SystemData){
        for (ui_transform, mut text) in (&ui_transforms, &mut texts).join() {
            match &*ui_transform.id {
                "timer" => {
                    text.text = timer.get_text();
                }
                "pb" => {}
                "wr" => {}
                "segment" => {
                    text.text = runtime_progress.current_segment.to_string();
                }
                "speed" => {
                    for (_, velocity) in (&players, &velocities).join() {
                        let vel = velocity.linear();
                        let vel_flat = Vector3::new(vel.x, 0.0, vel.z);
                        let mag = vel_flat.magnitude() * DISPLAY_SPEED_MULTIPLIER;

                        text.text = avg_float_to_string(mag, 1);
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn avg_float_to_string(value: f32, decimals: u32) -> String {
    let mult = 10.0_f32.powf(decimals as f32);
    ((value * mult).ceil() / mult).to_string()
}

pub fn add_removal_to_entity(entity: Entity, id: RemovalId, world: &World) {
	world
        .write_storage::<Removal<RemovalId>>()
        .insert(entity, Removal::new(id))
        .expect(&format!("Failed to insert removalid to entity {:?}.", entity));
}

#[derive(Default)]
struct InitState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for InitState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.register::<ObjectType>();
        data.world.register::<Removal<RemovalId>>();
        data.world.add_resource(get_all_maps(&get_working_dir()));
        data.world.add_resource(AmbientColor(Rgba::from([0.1; 3])));

        let mut world_param = WorldParameters::new(-Vector3::<f32>::unit_y());
        world_param = world_param.with_damping(1.0);
        data.world.add_resource(world_param);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::Switch(Box::new(MainMenuState))
    }
}

#[derive(Default)]
struct MainMenuState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for MainMenuState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let hide_cursor = HideCursor { hide: false };
        data.world.add_resource(hide_cursor);

        data.world.register::<ObjectType>();
        data.world.register::<Removal<RemovalId>>();

        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/menu_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::MenuUi, &data.world);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "play_button" => Trans::Switch(Box::new(MapSelectState::default())),
                        "quit_button" => Trans::Quit,
                        _ => Trans::None,
                    }
                } else {
                    Trans::None
                }
            }
            _ => Trans::None,
        }
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::MenuUi,
        );
    }
}

#[derive(Default)]
struct MapSelectState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for MapSelectState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data.world.exec(|mut creator: UiCreator| {
            creator.create("assets/base/prefabs/map_select_ui.ron", ())
        });
        add_removal_to_entity(ui_root, RemovalId::MapSelectUi, &data.world);

        let font = data
            .world
            .read_resource::<AssetLoader>()
            .load(
                "font/arial.ttf",
                FontFormat::Ttf,
                (),
                &mut data.world.write_resource(),
                &mut data.world.write_resource(),
                &data.world.read_resource(),
            ).expect("Failed to load font");
        let maps = data.world.read_resource::<MapInfoCache>().maps.clone();
        for (accum, (internal, info)) in maps.into_iter().enumerate() {
            info!("adding map!");
            let entity =
                UiButtonBuilder::new(format!("map_select_{}", internal), info.name.clone())
                    .with_font(font.clone())
                    .with_text_color([0.2, 0.2, 0.2, 1.0])
                    .with_font_size(30.0)
                    .with_size(512.0, 200.0)
                    .with_layer(8.0)
                    .with_position(0.0, -300.0 - 100.0 * accum as f32)
                    .with_anchor(Anchor::TopMiddle)
                    .build_from_world(data.world);
            add_removal_to_entity(entity, RemovalId::MapSelectUi, &data.world);
        }
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        let mut change_map = None;
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "back_button" => {
                            return Trans::Switch(Box::new(MainMenuState::default()));
                        }
                        id => {
                            if id.starts_with("map_select_") {
                                let map_name = &id[11..];
                                change_map = Some(
                                    data.world
                                        .read_resource::<MapInfoCache>()
                                        .maps
                                        .iter()
                                        .find(|t| t.0 == map_name)
                                        .unwrap()
                                        .clone(),
                                );
                            }
                        }
                    }
                }
            }
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    return Trans::Switch(Box::new(MainMenuState::default()));
                }
            }
            _ => {}
        }

        if let Some(row) = change_map {
            data.world.add_resource(CurrentMap::new(row.0, row.1));
            return Trans::Switch(Box::new(MapLoadState::default()));
        }
        Trans::None
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::MapSelectUi,
        );
    }
}

#[derive(Default)]
struct GameplayState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for GameplayState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = true;
        data.world.write_resource::<Time>().set_time_scale(1.0);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        time_sync(&data.world);
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        _data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        // TODO: Map finished
        match event {
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    Trans::Push(Box::new(PauseMenuState::default()))
                } else {
                    Trans::None
                }
            }
            StateEvent::Custom(CustomStateEvent::GotoMainMenu) => {
                Trans::Switch(Box::new(MapSelectState::default()))
            }
            StateEvent::Custom(CustomStateEvent::MapFinished) => {
                Trans::Switch(Box::new(ResultState::default()))
            }
            _ => Trans::None,
        }
    }

    fn on_pause(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = false;
        data.world.write_resource::<Time>().set_time_scale(0.0);
    }

    fn on_resume(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = true;
        data.world.write_resource::<Time>().set_time_scale(1.0);
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = false;
        // Not sure if I should put 0. Might cause errors later when implementing replays and stuff.
        data.world.write_resource::<Time>().set_time_scale(1.0);
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::Scene,
        );
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::GameplayUi,
        );
    }
}

#[derive(Default)]
struct PauseMenuState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for PauseMenuState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/pause_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::PauseUi, &data.world);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        // Necessary otherwise rhusics will keep the same DeltaTime and will not be paused.
        time_sync(&data.world);
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "resume_button" => Trans::Pop,
                        "quit_button" => {
                            data.world
                                .write_resource::<EventChannel<CustomStateEvent>>()
                                .single_write(CustomStateEvent::GotoMainMenu);
                            Trans::Pop
                        }
                        _ => Trans::None,
                    }
                } else {
                    Trans::None
                }
            }
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    Trans::Pop
                } else {
                    Trans::None
                }
            }
            _ => Trans::None,
        }
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::PauseUi,
        );
    }
}


pub fn verts_from_mesh_data(mesh_data: &MeshData, scale: &Vector3<f32>) -> Vec<Point3<f32>> {
	if let MeshData::Creator(combo) = mesh_data {
        combo
            .vertices()
            .iter()
            .map(|sep| {
                Point3::new(
                    (sep.0)[0] * scale.x,
                    (sep.0)[1] * scale.y,
                    (sep.0)[2] * scale.z,
                )
            }).collect::<Vec<_>>()
    } else {
    	error!("MeshData was not of combo type! Not extracting vertices.");
        vec![]
    }
}


#[derive(Default)]
struct MapLoadState {
    load_progress: Option<ProgressCounter>,
    player_entity: Option<Entity>,
    init_done: bool,
}

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for MapLoadState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<Time>().set_time_scale(0.0);
        self.init_done = false;

        let mut pg = ProgressCounter::new();

        let name = data.world.read_resource::<CurrentMap>().0.clone();

        let scene_handle = data.world.exec(|loader: PrefabLoader<GltfPrefab>| {
            loader.load(
                gltf_path_from_map(&get_working_dir(), &name),
                GltfSceneFormat,
                GltfSceneOptions::default(),
                &mut pg,
            )
        });

        /*let player_data_handle = data.world.exec(|loader: PrefabLoader<PlayerPrefabData>| {
            loader.load(
                "assets/base/config/player.ron",
                RonFormat,
                (),
                &mut pg,
            )
        });*/

        let player_settings_data = std::fs::read_to_string(format!("{}/assets/base/config/player.ron",get_working_dir())).expect("Failed to read player.ron settings file.");
        let player_settings: PlayerSettings = ron::de::from_str(&player_settings_data).expect("Failed to load player settings from file.");

        self.load_progress = Some(pg);

        let scene_root = data.world.create_entity().with(scene_handle).build();
        add_removal_to_entity(scene_root, RemovalId::Scene, &data.world);

        data.world.add_resource(Gravity::new(Vector3::new(0.0, player_settings.gravity, 0.0)));

        let player_entity = data
            .world
            .create_entity()
            .with(player_settings.movement) 
            .with(player_settings.grounded)
            .with(player_settings.ground_friction)
            .with(FlyControlTag)
            .with(ObjectType::Player)
            .with(Jump::new(true, true, player_settings.jump_velocity, true))
            .with(Player)
            .with_dynamic_physical_entity(
                Shape::new_simple_with_type(
                    CollisionStrategy::FullResolution,
                    CollisionMode::Discrete,
                    //Cylinder::new(0.5, 0.2).into(),
                    player_settings.shape.clone(),
                    ObjectType::Player,
                ),
                BodyPose3::new(
                    Point3::new(0.,0.,0.),
                    Quaternion::<f32>::one(),
                ),
                Velocity3::default(),
                player_settings.physical_entity.clone(),
                Mass3::new(player_settings.mass),
                //player_settings.mass.clone(),
            )
            .with(Transform::default())
            .with(ForceAccumulator::<Vector3<f32>, Vector3<f32>>::new())
            .with(Removal::new(RemovalId::Scene))
            .build();

        let mut tr = Transform::default();
        
        // TODO add conf ability to this
        tr.translation = [0.0, 0.25, 0.0].into();
        data.world
            .create_entity()
            .with(tr)
            .with(RotationControl::default())
            .with(Camera::standard_3d(1920.0, 1080.0))
            .with(Parent {
                entity: player_entity.clone(),
            }).build();

        self.player_entity = Some(player_entity);

        let mut tr = Transform::default();
        tr.translation = [0.0, 10.0, 0.0].into();
        tr.rotation = Quaternion::one();
        
        data.world
            .create_entity()
            .with(tr)
            .with(Light::Point(PointLight {
                color: Rgba::white(),
                intensity: 10.0,
                radius: 60.0,
                smoothness: 4.0,
            })).with(Removal::new(RemovalId::Scene))
            .build();

        let ui_root = data.world.exec(|mut creator: UiCreator| {
            creator.create("assets/base/prefabs/gameplay_ui.ron", ())
        });
        add_removal_to_entity(ui_root, RemovalId::GameplayUi, &data.world);

        data.world.add_resource(RuntimeProgress::default());
        MyPhysicalEntityParts::setup(&mut data.world.res)
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        if !self.init_done && self.load_progress.as_ref().unwrap().is_complete() {
            info!("BBBBBBBBBBBBBBBBBBBBBBBBBBBBB");
            let entity_sizes = (
                &*data.world.entities(),
                &data.world.read_storage::<Transform>(),
                &data.world.read_storage::<MeshData>(),
                &data.world.read_storage::<Named>(),
            )
                .par_join()
                .map(|(entity, transform, mesh_data, name)| {
                    let verts = verts_from_mesh_data(mesh_data, &transform.scale);
                    (entity, transform.clone(), verts, name.name.clone())
                }).collect::<Vec<_>>();

            if !entity_sizes.is_empty() {
                // The loading is done, now we add the colliders.

                let mut runtime_map = RuntimeMapBuilder::default();

                let mut max_segment = 0;
                self.init_done = true;

                let max_segment = {
                	let (mut physical_parts, mut object_types, players) = <(MyPhysicalEntityParts, WriteStorage<ObjectType>, ReadStorage<Player>) as SystemData>::fetch(&data.world.res);
                    for (entity, transform, mesh, name) in entity_sizes {
                        let (obj_type, coll_strat) = if name == "StartZone" {
                            // Move player to StartZone
		                    for (mut body_pose, _) in (&mut physical_parts.next_poses, &players).join() {
		                        body_pose.value.set_position(Point3::new(transform.translation.x, transform.translation.y, transform.translation.z));
		                        body_pose.value.set_rotation(transform.rotation);
		                    }
		                    if runtime_map.start_zone.is_some() {
		                    	panic!("There can be only one StartZone per map");
		                    }
		                    runtime_map = runtime_map.start_zone(entity);
                            (ObjectType::StartZone, CollisionStrategy::CollisionOnly)
                        } else if name == "EndZone" {
                        	if runtime_map.end_zone.is_some() {
		                    	panic!("There can be only one EndZone per map");
		                    }
                        	runtime_map = runtime_map.end_zone(entity);
                            (ObjectType::EndZone, CollisionStrategy::CollisionOnly)
                        } else if name.starts_with("KillZone") {
                            (ObjectType::KillZone, CollisionStrategy::CollisionOnly)
                        } else if name.starts_with("SegmentZone") {
                            let id_str = &name[11..];
                            let id = id_str.to_string().parse::<u8>().unwrap(); // TODO error handling for maps
                            if id > max_segment {
                                max_segment = id;
                            }

                            runtime_map.segment_zones.push((id,entity));
                            (ObjectType::SegmentZone(id),CollisionStrategy::CollisionOnly)
                        } else {
                            (ObjectType::Scene, CollisionStrategy::FullResolution)
                        };

                        object_types
                            .insert(entity, obj_type.clone())
                            .expect("Failed to add ObjectType to map mesh");
                        physical_parts
                            .static_entity(
                                entity,
                                Shape::new_simple_with_type(
                                    coll_strat,
                                    CollisionMode::Discrete,
                                    Primitive3::ConvexPolyhedron(<ConvexPolyhedron<f32>>::new(
                                        mesh,
                                    )),
                                    obj_type,
                                ),
                                BodyPose3::new(
                                    Point3::new(
                                        transform.translation.x,
                                        transform.translation.y,
                                        transform.translation.z,
                                    ),
                                    transform.rotation,
                                ),
                                PhysicalEntity::default(),
                                Mass3::infinite(),
                            ).expect("Failed to add static collider to map mesh");
                    }

                	max_segment
            	};

            	// Validate map
            	runtime_map.build(&data.world.read_resource::<CurrentMap>().1).unwrap();


                data.world.add_resource(RuntimeProgress::new(max_segment));
            }
        } else if self.init_done {
            return Trans::Switch(Box::new(GameplayState::default()));
        }

        data.data.update(&data.world);
        Trans::None
    }
}

#[derive(Default)]
struct ResultState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for ResultState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/result_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::ResultUi, &data.world);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "back_button" => Trans::Switch(Box::new(MapSelectState::default())),
                        _ => Trans::None,
                    }
                } else {
                    Trans::None
                }
            }
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    Trans::Switch(Box::new(MapSelectState::default()))
                } else {
                    Trans::None
                }
            }
            _ => Trans::None,
        }
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::ResultUi,
        );
    }
}

fn main() -> amethyst::Result<()> {
    /*


    unique maps to be more than a cs port
    pushed by triggered explosion

    maps totally reset to origin when restarting, thus allowing for cool runtime effects
    possible puzzles?

    hidden doors
    levers activating hidden doors later in the level
    


    */

    amethyst::start_logger(Default::default());

    let resources_directory = get_working_dir();

    let asset_loader = AssetLoader::new(&format!("{}/assets", resources_directory), "base");

    let display_config_path = asset_loader.resolve_path("config/display.ron").unwrap();
    let display_config = DisplayConfig::load(&display_config_path);

    let key_bindings_path = asset_loader.resolve_path("config/input.ron").unwrap();

    let pipe = Pipeline::build().with_stage(
        Stage::with_backbuffer()
            .clear_target([0.1, 0.1, 0.1, 1.0], 1.0)
            .with_pass(DrawPbmSeparate::new().with_transparency(
                ColorMask::all(),
                ALPHA,
                Some(DepthMode::LessEqualWrite),
            )).with_pass(DrawUi::new()),
    );

    let game_data = GameDataBuilder::default()
        .with(RelativeTimerSystem, "relative_timer", &[])
        .with(
            PrefabLoaderSystem::<ScenePrefab>::default(),
            "map_loader",
            &[],
        )
        /*.with(
            PrefabLoaderSystem::<PlayerPrefabData>::default(),
            "player_loader",
            &[],
        )*/
        .with(
            GltfSceneLoaderSystem::default(),
            "gltf_loader",
            &["map_loader"],
        ).with(
            FPSRotationRhusicsSystem::<String, String>::new(0.3, 0.3),
            "free_rotation",
            &[],
        ).with(MouseFocusUpdateSystem::new(), "mouse_focus", &[])
        .with(CursorHideSystem::new(), "cursor_hide", &[])
        .with(GroundCheckerSystem::new(vec![ObjectType::Scene]), "ground_checker", &[])
        .with(JumpSystem::default(), "jump", &["ground_checker"])
        .with(
            GroundFrictionSystem,
            "ground_friction",
            &["ground_checker", "jump"],
        ).with(
            BhopMovementSystem::<String, String>::new(
                Some(String::from("right")),
                Some(String::from("forward")),
            ),
            "bhop_movement",
            &["free_rotation", "jump", "ground_friction", "ground_checker"],
        )
        .with(GravitySystem, "gravity", &[])
        .with_bundle(TransformBundle::new().with_dep(&[]))?
        .with(ContactSystem::default(), "contacts", &["bhop_movement"])
        .with(UiUpdaterSystem, "gameplay_ui_updater", &[])
        //.with(UiAutoTextSystem::<RelativeTimer>::default(), "timer_text_update", &[])
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?.with_bundle(UiBundle::<String, String>::new())?
        .with_barrier()
        .with_bundle(DefaultPhysicsBundle3::<ObjectType>::new().with_spatial())?
        .with_bundle(RenderBundle::new(pipe, Some(display_config)))?;
    let mut game = Application::build(resources_directory, InitState::default())?
        .with_resource(asset_loader)
        .with_resource(AssetLoaderInternal::<FontAsset>::new())
        .build(game_data)?;
    game.run();
    Ok(())
}
