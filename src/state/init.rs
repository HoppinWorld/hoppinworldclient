
use util::get_all_maps;
use amethyst::utils::application_root_dir;
use amethyst::renderer::*;
use amethyst::controls::HideCursor;
use amethyst::core::nalgebra::Vector3;
use state::login::LoginState;
use amethyst::prelude::*;
use amethyst::utils::removal::*;
use hoppinworldruntime::{PlayerSettings, ObjectType, AllEvents, CustomTrans, RemovalId};
use tokio::runtime::Runtime;

#[derive(Default)]
pub struct InitState;

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for InitState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.register::<ObjectType>();
        data.world.register::<Removal<RemovalId>>();
        data.world.add_resource(get_all_maps(&application_root_dir().unwrap().to_str().unwrap()));
        data.world.add_resource(AmbientColor(Rgba::from([0.1; 3])));
        let hide_cursor = HideCursor { hide: false };
        data.world.add_resource(hide_cursor);

        let player_settings_data = std::fs::read_to_string(format!("{}/assets/base/config/player.ron",application_root_dir().unwrap().to_str().unwrap())).expect("Failed to read player.ron settings file.");
        let player_settings: PlayerSettings = ron::de::from_str(&player_settings_data).expect("Failed to load player settings from file.");

        data.world.add_resource(player_settings);

        //let mut runtime = Arc::new(Mutex::new(Runtime::new().expect("Failed to create tokio runtime")));
        let runtime = Runtime::new().expect("Failed to create tokio runtime");
        data.world.add_resource(runtime);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::Switch(Box::new(LoginState))
        //Trans::Switch(Box::new(MainMenuState))
    }
}