
use amethyst_extra::set_discord_state;
use tokio::runtime::Runtime;
use {add_removal_to_entity, do_login, Auth};
use amethyst::prelude::*;
use amethyst::utils::removal::*;
use amethyst::ui::*;
use amethyst::ecs::SystemData;
use state::*;
use hoppinworldruntime::{AllEvents, CustomTrans, RemovalId};

#[derive(Default)]
pub struct LoginState;

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for LoginState {
    fn on_start(&mut self, mut data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/login_ui.ron", ()));
        add_removal_to_entity(ui_root, RemovalId::LoginUi, &data.world);

        set_discord_state(String::from("Login"), &mut data.world);
    }

    fn update(&mut self, mut data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);

        if let Some(_) = data.world.res.try_fetch::<Auth>() {
            return Trans::Switch(Box::new(MainMenuState::default()));
        }

        /*while let Some(f) = data.world.write_resource::<FutureProcessor>().queue.lock().unwrap().pop_front() {
            f(&mut data.world);
        }*/
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
                if let Some(ui_transform_id) = data.world.read_storage::<UiTransform>().get(entity).map(|tr| tr.id.clone()) {
                    match &*ui_transform_id {
                        "login_button" => {
                            let username_entity = UiFinder::fetch(&data.world.res).find("username").unwrap();
                            let username = data.world.read_storage::<UiText>().get(username_entity).unwrap().text.clone();
                            let password_entity = UiFinder::fetch(&data.world.res).find("password").unwrap();
                            let password = data.world.read_storage::<UiText>().get(password_entity).unwrap().text.clone();
                            do_login(&mut data.world.write_resource::<Runtime>(), data.world.read_resource::<CallbackQueue>().send_handle(), username, password);
                            Trans::None
                        },
                        "guest_button" => Trans::Switch(Box::new(MainMenuState::default())),
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
            RemovalId::LoginUi,
        );
    }
}