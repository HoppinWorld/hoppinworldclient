use resource::CurrentMap;
use hoppinworldruntime::{AllEvents, RuntimeProgress, CustomTrans, RemovalId};
use amethyst_extra::AssetLoader;
use {add_removal_to_entity, Auth, sec_to_display, submit_score, ScoreInsertRequest};
use amethyst::ui::{UiCreator, UiTransform, UiText, UiEvent, Anchor, FontFormat, UiFinder, UiEventType};
use amethyst::utils::removal::{exec_removal, Removal};
use amethyst::input::is_key_down;
use amethyst::prelude::*;
use amethyst::ecs::SystemData;
use amethyst::renderer::VirtualKeyCode;
use state::MapSelectState;

#[derive(Default)]
pub struct ResultState {
    finished: bool,
}

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for ResultState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("base/prefabs/result_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::ResultUi, &data.world);

        // Time table.
        let runtime_progress = data.world.read_resource::<RuntimeProgress>().clone();

        info!("SEGMENT TIMES: ");
        for t in &runtime_progress.segment_times {
            print!("{},", t);
        }
        info!("");
        info!("RUN DONE!");

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

        let mut accum = 0.0;
        for (segment, time) in runtime_progress.segment_times.iter().enumerate() {
            // Accum
            data.world.create_entity()
                .with(UiTransform::new(String::from(""), Anchor::TopMiddle, -200.0, -350.0 - 100.0 * segment as f32, 3.0, 200.0, 100.0, -1))
                .with(UiText::new(font.clone(), sec_to_display(*time, 3), [0.1,0.1,0.1,1.0], 35.0))
                .with(Removal::new(RemovalId::ResultUi))
                .build();


            let diff = if *time == 0.0{
                0.0
            } else {
                *time - accum
            };
            if *time != 0.0 {
                accum = *time;
            }

            // Segment
            data.world.create_entity()
                .with(UiTransform::new(String::from(""), Anchor::TopMiddle, 200.0, -350.0 - 100.0 * segment as f32, 3.0, 200.0, 100.0, -1))
                .with(UiText::new(font.clone(), sec_to_display(diff, 3), [0.1,0.1,0.1,1.0], 35.0))
                .with(Removal::new(RemovalId::ResultUi))
                .build();
        }

        // Web submit score if logged in
        if let Some(auth_token) = data.world.res.try_fetch::<Auth>().map(|a| a.token.clone()) {
            let times = runtime_progress.segment_times.iter().map(|f| *f as f32);
            let total_time = runtime_progress.segment_times.iter().map(|f| *f as f32).last().unwrap();
            let insert = ScoreInsertRequest {
                mapid: 1,
                segment_times: times.collect(),
                strafes: 0,
                jumps: 0,
                total_time: total_time,
                max_speed: 0.0,
                average_speed: 0.0,
            };

            submit_score(&mut data.world.write_resource(), auth_token, insert);
        }
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);

        if !self.finished {
            // Set the map name
            if let Some(map_name_entity) = UiFinder::fetch(&data.world.res).find("map_name") {
                let map_name = data.world.read_resource::<CurrentMap>().1.name.clone();
                data.world.write_storage::<UiText>().get_mut(map_name_entity).unwrap().text = map_name;
                self.finished = true;
            }
        }

        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: AllEvents,
    ) -> CustomTrans<'a, 'b> {
        match event {
            AllEvents::Ui(UiEvent {
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
            AllEvents::Window(ev) => {
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