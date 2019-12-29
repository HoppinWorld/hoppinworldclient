use add_removal_to_entity;
use amethyst::prelude::*;
use amethyst::ui::*;
use amethyst::utils::removal::*;
use amethyst::core::Time;
use amethyst_extra::set_discord_state;
use hoppinworld_runtime::{AllEvents, CustomTrans, RemovalId};
use state::*;

#[derive(Default)]
pub struct MainMenuState;

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for MainMenuState {
    fn on_start(&mut self, mut data: StateData<GameData>) {
        data.world.write_resource::<Time>().set_time_scale(0.0);

        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("base/prefabs/menu_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::MenuUi, &mut data.world);

        set_discord_state(String::from("Main Menu"), &mut data.world);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(&mut self, data: StateData<GameData>, event: AllEvents) -> CustomTrans<'a, 'b> {
        match event {
            AllEvents::Ui(UiEvent {
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
