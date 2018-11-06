
use amethyst_rhusics::collision::primitive::{ConvexPolyhedron, Cylinder, Primitive3};
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    CollisionMode, CollisionStrategy, ForceAccumulator,
    PhysicalEntity, Pose
};
use amethyst_rhusics::rhusics_ecs::physics3d::BodyPose3;
use amethyst_rhusics::rhusics_ecs::WithPhysics;
use amethyst::ecs::*;
use amethyst::ecs::prelude::ParallelIterator;
use amethyst::assets::{ProgressCounter, Handle};
use amethyst_extra::*;
use util::gltf_path_from_map;
use amethyst_gltf::{GltfSceneFormat, GltfSceneOptions};
use amethyst::controls::FlyControlTag;
use hoppinworldruntime::{PlayerTag, PlayerSettings, PlayerFeetTag, Shape};
use amethyst::core::cgmath::{Vector3, Quaternion, Point3, One};
use amethyst::core::*;
use amethyst::renderer::*;
use verts_from_mesh_data;
use resource::CurrentMap;
use hoppinworldruntime::{MyPhysicalEntityParts, ObjectType, AllEvents, RuntimeProgress, CustomTrans, RemovalId, RuntimeMapBuilder};
use amethyst_extra::AssetLoader;
use {add_removal_to_entity, Gravity};
use amethyst::ui::UiCreator;
use amethyst::utils::removal::Removal;
use amethyst::prelude::*;
use state::{MapSelectState, GameplayState};
use partial_function::PartialFunctionBuilder;

#[derive(Default)]
pub struct MapLoadState {
    load_progress: Option<ProgressCounter>,
    player_entity: Option<Entity>,
    init_done: bool,
}

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for MapLoadState {
    fn on_start(&mut self, mut data: StateData<GameData>) {
        data.world.write_resource::<Time>().set_time_scale(0.0);

        self.init_done = false;

        let pg = ProgressCounter::new();

        let name = data.world.read_resource::<CurrentMap>().0.clone();
        let display_name = data.world.read_resource::<CurrentMap>().1.name.clone();

        set_discord_state(format!("Hoppin On: {}", display_name.clone()), &mut data.world);

        let scene_handle = data.world.read_resource::<AssetLoader>()
            .load(
                //&format!("../../{}",gltf_path_from_map(&get_working_dir(), &name)),
                &gltf_path_from_map("../..", &name),
                GltfSceneFormat,
                GltfSceneOptions{
                    flip_v_coord: true,
                    ..Default::default()
                },
                &mut data.world.write_resource(),
                &mut data.world.write_resource(),
                &data.world.read_resource(),
        );

        /*let scene_handle = data.world.exec(|loader: PrefabLoader<GltfPrefab>| {
            loader.load(
                gltf_path_from_map(&get_working_dir(), &name),
                GltfSceneFormat,
                GltfSceneOptions{
                    flip_v_coord: true,
                    ..Default::default()
                },
                &mut pg,
            )
        });*/

        if let None = scene_handle {
            error!("Failed to load map!");
            return;
        }

        let player_settings = data.world.read_resource::<PlayerSettings>().clone();

        self.load_progress = Some(pg);

        data.world.create_entity()
            .with(scene_handle.unwrap())
            .with(Removal::new(RemovalId::Scene))
            .build();

        data.world.add_resource(Gravity::new(Vector3::new(0.0, player_settings.gravity, 0.0)));

        let mut jump = Jump::new(true, true, player_settings.jump_velocity, true);
        jump.jump_timing_boost = Some(
            PartialFunctionBuilder::new().with(-0.20, 0.20, |t| {
                let max = 0.25;
                if t == 0.0 {
                    1.0 + max as f32
                } else {
                    1.0 + (0.005_f64/t).abs().max(max) as f32
                }
            }).build()
        );

        let player_entity = data
            .world
            .create_entity()
            .with(player_settings.movement) 
            .with(player_settings.grounded)
            .with(player_settings.ground_friction)
            .with(FlyControlTag)
            .with(ObjectType::Player)
            .with(jump)
            .with(PlayerTag)
            .with_dynamic_physical_entity(
                Shape::new_simple_with_type(
                    CollisionStrategy::FullResolution,
                    CollisionMode::Discrete,
                    //CollisionMode::Continuous,
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


        // Secondary ground collider
        data.world
            .create_entity()
            .with(ObjectType::PlayerFeet)
            .with_static_physical_entity(
                Shape::new_simple_with_type(
                    CollisionStrategy::CollisionOnly,
                    //CollisionStrategy::FullResolution,
                    CollisionMode::Discrete,
                    Cylinder::new(0.01, 0.155).into(),
                    ObjectType::PlayerFeet,
                ),
                BodyPose3::new(
                    Point3::new(0., 0., 0.),
                    Quaternion::<f32>::one(),
                ),
                PhysicalEntity::default(),
                Mass3::infinite(),
                //player_settings.mass.clone(),
            )
            .with(PlayerFeetTag)
            .with(Transform::default())
            .with(Removal::new(RemovalId::Scene))
            .build();

        // Assign secondary collider to player's Grounded component
        //(&mut data.world.write_storage::<Grounded>()).join().for_each(|grounded| grounded.watch_entity = Some(secondary.clone()));

        self.player_entity = Some(player_entity);

        let mut tr = Transform::default();
        tr.translation = [0.0, 10.0, 0.0].into();
        tr.rotation = Quaternion::one();
        
        /*data.world
            .create_entity()
            .with(tr)
            .with(Light::Sun(SunLight {
                ang_rad: 10.0,
                color: Rgba::white(),
                direction: [0.1, -1.0, 0.05],
                intensity: 10.0,
            })).with(Removal::new(RemovalId::Scene))
            .build();*/

        data.world
            .create_entity()
            .with(tr)
            .with(Light::Directional(DirectionalLight {
                color: [1.0, 1.0, 1.0, 1.0].into(),
                direction: [0.1, -1.0, 0.05],
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
        if self.load_progress.is_none() {
            return Trans::Switch(Box::new(MapSelectState::default()));
        }
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
                	let (mut physical_parts, mut object_types, mut meshes, players, mut removals) = <(MyPhysicalEntityParts, WriteStorage<ObjectType>, WriteStorage<Handle<Mesh>>, ReadStorage<PlayerTag>, WriteStorage<Removal<RemovalId>>) as SystemData>::fetch(&data.world.res);
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
                        } else if name.starts_with("Ignore") {
                            (ObjectType::Ignore, CollisionStrategy::CollisionOnly)
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

                        if name.contains("Invisible") {
                            if meshes.contains(entity) {
                                meshes.remove(entity);
                            }
                        }

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
                        removals.insert(entity, Removal::new(RemovalId::Scene)).unwrap();
                    }

                	max_segment
            	};

            	// Validate map
            	runtime_map.build(&data.world.read_resource::<CurrentMap>().1).unwrap();


                data.world.add_resource(RuntimeProgress::new(max_segment + 1));
            }
        } else if self.init_done {
            return Trans::Switch(Box::new(GameplayState::default()));
        }

        data.data.update(&data.world);
        Trans::None
    }
}