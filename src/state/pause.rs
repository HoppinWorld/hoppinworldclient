use add_removal_to_entity;
use amethyst::input::*;
use amethyst::prelude::*;
use amethyst::renderer::VirtualKeyCode;
use amethyst::shrev::EventChannel;
use amethyst::ui::*;
use amethyst::utils::removal::*;
use hoppinworldruntime::{AllEvents, CustomStateEvent, CustomTrans, RemovalId};

#[derive(Default)]
pub struct PauseMenuState;

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for PauseMenuState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("base/prefabs/pause_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::PauseUi, &data.world);
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
                        "resume_button" => Trans::Pop,
                        "retry_button" => {
                            data.world
                                .write_resource::<EventChannel<CustomStateEvent>>()
                                .single_write(CustomStateEvent::Retry);
                            Trans::Pop
                        }
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
            AllEvents::Window(ev) => {
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
