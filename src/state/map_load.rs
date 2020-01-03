use amethyst_extra::nphysics_ecs::ncollide::shape::*;
use amethyst::core::ecs::prelude::ParallelIterator;
use amethyst::assets::{Handle, ProgressCounter};
use amethyst::controls::FlyControlTag;
use amethyst::core::math::{Isometry3, Matrix3, Point3, UnitQuaternion, Vector3};
use amethyst::core::*;
use amethyst::ecs::*;
use amethyst::prelude::*;
use amethyst::renderer::*;
use amethyst::renderer::rendy::mesh::PosTex;
use amethyst::ui::UiCreator;
use amethyst::utils::removal::Removal;
use amethyst_extra::nphysics_ecs::nphysics::object::*;
use amethyst_extra::nphysics_ecs::nphysics::material::*;
use amethyst_extra::nphysics_ecs::nphysics::volumetric::Volumetric;
use amethyst_extra::nphysics_ecs::*;
use amethyst_extra::AssetLoader;
use amethyst_extra::*;
use amethyst::gltf::{GltfSceneAsset, GltfSceneFormat, GltfSceneOptions, VerticeData};
use hoppinworld_runtime::{
    AllEvents, CustomTrans, ObjectType, RemovalId, RuntimeMapBuilder, RuntimeProgress,
};
use hoppinworld_runtime::{PlayerFeetTag, PlayerSettings, PlayerTag};
use num_traits::identities::One;
use partial_function::PartialFunctionBuilder;
use resource::CurrentMap;
use state::{GameplayState, MapSelectState};
use util::gltf_path_from_map;
use verts_from_mesh_data;
use add_removal_to_entity;

#[derive(Default)]
pub struct MapLoadState {
    load_progress: Option<ProgressCounter>,
    init_done: bool,
}

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for MapLoadState {
    fn on_start(&mut self, mut data: StateData<GameData>) {
        data.world.write_resource::<Time>().set_time_scale(0.0);

        self.init_done = false;

        let pg = ProgressCounter::new();

        let name = data.world.read_resource::<CurrentMap>().0.clone();
        let display_name = data.world.read_resource::<CurrentMap>().1.name.clone();

        let player_settings = (*data.world.read_resource::<PlayerSettings>()).clone();

        set_discord_state(
            format!("Hoppin On: {}", display_name.clone()),
            &mut data.world,
        );

        let scene_handle: Option<Handle<GltfSceneAsset>> = data.world.read_resource::<AssetLoader>().load(
            //&format!("../../{}",gltf_path_from_map(&get_working_dir(), &name)),
            &gltf_path_from_map("../..", &name),
            GltfSceneFormat(
                GltfSceneOptions {
                    flip_v_coord: true,
                    ..Default::default()
                }
            ),
            &mut data.world.write_resource(),
            &mut data.world.write_resource(),
            &data.world.read_resource(),
        );

        error!("Loading map from {}", &gltf_path_from_map("../..", &name));

        if let None = scene_handle {
            error!("Failed to load map!");
            return;
        }


        self.load_progress = Some(pg);

        data.world
            .create_entity()
            .with(scene_handle.unwrap())
            .with(Removal::new(RemovalId::Scene))
            .build();

        let physics_mat = MaterialHandle::new(BasicMaterial::new(0.0, 0.0));

        let mut jump = Jump::new(true, true, player_settings.jump_velocity, true);
        jump.jump_timing_boost = Some(
            PartialFunctionBuilder::new()
                .with(-0.20, 0.20, |t| {
                    let max = 0.25;
                    if t == 0.0 {
                        1.0 + max as f32
                    } else {
                        1.0 + (0.005_f64 / t).abs().max(max) as f32
                    }
                })
                .build(),
        );

        let mut grounded = player_settings.grounded.clone();

        let player_scale = Vector3::new(0.2, 0.4, 0.2); // Half the desired size
        let player_rotation =
            UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        let s = amethyst::renderer::shape::Shape::Cylinder(12, None)
            .generate_vertices::<Vec<PosTex>>(None)
            .into_iter()
            .map(|mut pt| {
                let mut p = player_rotation * Point3::from(pt.position.0);
                p.x *= player_scale.x;
                p.y *= player_scale.y;
                p.z *= player_scale.z;
                p
            })
            .collect::<Vec<_>>();
        let s2 = amethyst::renderer::shape::Shape::Cylinder(12, None)
            .generate_vertices::<Vec<PosTex>>(None)
            .into_iter()
            .map(|mut pt| {
                let mut p = player_rotation * Point3::from(pt.position.0);
                p.x *= player_scale.x * 0.95;
                p.y *= 0.1;
                p.z *= player_scale.z * 0.95;
                p
            })
            .collect::<Vec<_>>();
        //let shape = ShapeHandle::new(Cylinder::new(0.4, 0.2));
        //let shape = ShapeHandle::new(Ball::new(0.2));
        let shape = ShapeHandle::new(
            ConvexHull::try_from_points(&s)
                .expect("Failed to create player collision hull from points"),
        );
        let feet_shape = ShapeHandle::new(
            ConvexHull::try_from_points(&s2)
                .expect("Failed to create player feet collision hull from points"),
        );
        let player_entity = data
            .world
            .create_entity()
            .with(player_settings.movement.clone())
            .with(player_settings.ground_friction.clone())
            .with(FlyControlTag)
            .with(ObjectType::Player)
            .with(jump)
            .with(PlayerTag)
            .with(BodyComponent::new(
                    RigidBodyDesc::new()
                        .mass(player_settings.mass)
                        .local_center_of_mass(shape.center_of_mass())
                        .build()
            ))
            .with(Transform::default())
            .with(Removal::new(RemovalId::Scene))
            .build();

        data.world.write_storage::<ColliderComponent<f32>>().insert(player_entity, 
            ColliderComponent(ColliderDesc::new(shape)
                .set_collision_groups(ObjectType::Player.into()) // Player
                .set_material(physics_mat.clone())
                .build(BodyPartHandle(player_entity, 0))));

        // Create the main camera
        let tr = Transform::from(Vector3::new(0.0, 0.25, 0.0));
        data.world
            .create_entity()
            .with(tr)
            .with(RotationControl::default())
            .with(Camera::standard_3d(1920.0, 1080.0))
            .with(Removal::new(RemovalId::Scene))
            .with(Parent {
                entity: player_entity.clone(),
            })
            .build();

        // Secondary ground collider
        let ground_collider = data
            .world
            .create_entity()
            .with(ObjectType::PlayerFeet)
            .with(PlayerFeetTag)
            .with(Transform::default())
            .with(Removal::new(RemovalId::Scene))
            .build();

        data.world.write_storage::<ColliderComponent<f32>>().insert(ground_collider, 
            ColliderComponent(ColliderDesc::new(feet_shape)
                .set_collision_groups(ObjectType::PlayerFeet.into()) // Player Feet
                .set_material(physics_mat.clone())
                .set_is_sensor(true)
                .build(BodyPartHandle(ground_collider, 0))));

        // Assign secondary collider to player's Grounded component
        grounded.watch_entity = Some(ground_collider);
        data.world
            .write_storage::<Grounded>()
            .insert(player_entity, grounded)
            .unwrap();

        // Create ui
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("base/prefabs/gameplay_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::GameplayUi, &mut data.world);

        // Reset runtime progress
        data.world.insert(RuntimeProgress::default());
    }

    fn update(&mut self, mut data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        if self.load_progress.is_none() {
            return Trans::Switch(Box::new(MapSelectState::default()));
        }
        if !self.init_done && self.load_progress.as_ref().unwrap().is_complete() {
            info!("Map Prefab Loading");
            let entity_sizes = (
                &*data.world.entities(),
                &data.world.read_storage::<Transform>(),
                &data.world.read_storage::<VerticeData>(),
                &data.world.read_storage::<Named>(),
            )
                .par_join()
                .map(|(entity, transform, mesh_data, name)| {
                    let verts = verts_from_mesh_data(mesh_data, &transform.scale());
                    (entity, transform.clone(), verts, name.name.clone())
                })
                .collect::<Vec<_>>();

            if !entity_sizes.is_empty() {
                // The loading is done, now we add the colliders.

                let mut runtime_map = RuntimeMapBuilder::default();

                let mut max_segment = 0;
                self.init_done = true;

                let max_segment = {
                    let (
                        mut transforms,
                        mut colliders,
                        mut object_types,
                        mut meshes,
                        players,
                        mut removals,
                        mut ground_checks,
                    ) = <(
                        WriteStorage<Transform>,
                        WriteStorage<ColliderComponent<f32>>,
                        WriteStorage<ObjectType>,
                        WriteStorage<Handle<Mesh>>,
                        ReadStorage<PlayerTag>,
                        WriteStorage<Removal<RemovalId>>,
                        WriteStorage<GroundCheckTag>,
                    ) as SystemData>::fetch(&mut data.world);

                    let mut start_zone_pos = None;
                    let mut start_zone_rotation = None;

                    for (entity, transform, mesh, name) in entity_sizes {
                        let (obj_type, is_sensor) = if name == "StartZone" {
                            // Move player to StartZone
                            start_zone_pos = Some(transform.translation().clone());
                            start_zone_rotation = Some(transform.rotation().clone());

                            if runtime_map.start_zone.is_some() {
                                panic!("There can be only one StartZone per map");
                            }

                            runtime_map = runtime_map.start_zone(entity);

                            (ObjectType::StartZone, true)
                        } else if name == "EndZone" {
                            if runtime_map.end_zone.is_some() {
                                panic!("There can be only one EndZone per map");
                            }

                            runtime_map = runtime_map.end_zone(entity);

                            (ObjectType::EndZone, true)
                        } else if name.starts_with("KillZone") {
                            (ObjectType::KillZone, true)
                        } else if name.starts_with("Ignore") {
                            (ObjectType::Ignore, true)
                        } else if name.starts_with("SegmentZone") {
                            let id_str = &name[11..];
                            let id = id_str.to_string().parse::<u8>().unwrap(); // TODO error handling for maps
                            if id > max_segment {
                                max_segment = id;
                            }

                            runtime_map.segment_zones.push((id, entity));
                            (ObjectType::SegmentZone(id), true)
                        } else {
                            (ObjectType::Scene, false)
                        };

                        if name.contains("Invisible") {
                            if meshes.contains(entity) {
                                meshes.remove(entity);
                            }
                        }

                        object_types
                            .insert(entity, obj_type)
                            .expect("Failed to add ObjectType to map mesh");

                        // TODO: duplicated code
                        let physics_mat = MaterialHandle::new(BasicMaterial::new(0.0, 0.0));

                        match ConvexHull::try_from_points(&mesh) {
                            Some(handle) => {
                                colliders
                                    .insert(
                                        entity,
                                        ColliderComponent(ColliderDesc::new(ShapeHandle::new(handle))
                                            .set_collision_groups(obj_type.into()) // Scene or zones
                                            .set_material(physics_mat.clone())
                                            .set_is_sensor(is_sensor)
                                            .build(BodyPartHandle(entity, 0)))
                                    )
                                    .expect("Failed to add Collider to map mesh");
                            }
                            None => error!("Non-Convex mesh in scene! Mesh: {:?}", mesh),
                        }
                        if obj_type == ObjectType::Scene {
                            ground_checks.insert(entity, GroundCheckTag).unwrap();
                        }
                        removals
                            .insert(entity, Removal::new(RemovalId::Scene))
                            .unwrap();
                    }

                    /*for (tr, _) in (&mut transforms, &players).join() {
                        *tr.translation_mut() = start_zone_pos.expect("No start zone in scene.");
                        *tr.rotation_mut() = start_zone_rotation.expect("No start zone in scene.");

                        // *tr.translation_mut() = Vector3::new(0.0, 10.0, 10.0);
                    }*/
                    panic!("FIX THIS TODO TODO TODO TODO ^^^^^^^^^^^^^^^^^");

                    max_segment
                };

                // Validate map
                runtime_map
                    .build(&data.world.read_resource::<CurrentMap>().1)
                    .unwrap();

                data.world
                    .insert(RuntimeProgress::new(max_segment + 1));
            }
        } else if self.init_done {
            return Trans::Switch(Box::new(GameplayState::default()));
        }

        data.data.update(&data.world);
        Trans::None
    }
}
