use add_removal_to_entity;
use amethyst::input::*;
use amethyst::prelude::*;
use amethyst::input::VirtualKeyCode;
use amethyst::ui::*;
use amethyst::utils::removal::*;
use amethyst_extra::{set_discord_state, AssetLoader};
use hoppinworld_runtime::{AllEvents, CustomTrans, RemovalId};
use resource::{CurrentMap, MapInfoCache};
use state::*;

#[derive(Default)]
pub struct MapSelectState;

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for MapSelectState {
    fn on_start(&mut self, mut data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("base/prefabs/map_select_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::MapSelectUi, &mut data.world);

        let font = data
            .world
            .read_resource::<AssetLoader>()
            .load(
                "font/arial.ttf",
                TtfFormat,
                &mut data.world.write_resource(),
                &mut data.world.write_resource(),
                &data.world.read_resource(),
            )
            .expect("Failed to load font");
        let maps = data.world.read_resource::<MapInfoCache>().maps.clone();
        for (accum, (internal, info)) in maps.into_iter().enumerate() {
            info!("adding map!");
            let tup =
                UiButtonBuilder::<(), String>::new(info.name.clone())
                    .with_id(format!("map_select_{}", internal))
                    .with_font(font.clone())
                    .with_text_color([0.2, 0.2, 0.2, 1.0])
                    .with_font_size(30.0)
                    .with_size(512.0, 75.0)
                    .with_layer(8.0)
                    .with_position(0.0, -300.0 - 100.0 * accum as f32)
                    .with_anchor(Anchor::TopMiddle)
                    .build_from_world(data.world);
            add_removal_to_entity(tup.1.text_entity, RemovalId::MapSelectUi, &mut data.world);
        }

        set_discord_state(String::from("Main Menu"), &mut data.world);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(&mut self, data: StateData<GameData>, event: AllEvents) -> CustomTrans<'a, 'b> {
        let mut change_map = None;
        match event {
            AllEvents::Ui(UiEvent {
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
                                let len = map_name.len();
                                let map_name = &map_name[..len-4];
                                change_map = Some(
                                    data.world
                                        .read_resource::<MapInfoCache>()
                                        .maps
                                        .iter()
                                        .find(|t| t.0 == map_name)
                                        .expect(&format!("Map not found in cache for name {:?}", map_name))
                                        .clone(),
                                );
                            }
                        }
                    }
                }
            }
            AllEvents::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    return Trans::Switch(Box::new(MainMenuState::default()));
                }
            }
            _ => {}
        }

        if let Some(row) = change_map {
            data.world.insert(CurrentMap::new(row.0, row.1));
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
