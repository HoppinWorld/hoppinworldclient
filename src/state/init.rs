
use util::get_all_maps;
use amethyst_extra::get_working_dir;
use amethyst::renderer::*;
use amethyst::controls::HideCursor;
use amethyst::core::cgmath::Vector3;
use state::login::LoginState;
use FutureProcessor;
use amethyst::prelude::*;
use amethyst::utils::removal::*;
use hoppinworldruntime::{PlayerSettings, ObjectType, AllEvents, CustomTrans, RemovalId};
use tokio::runtime::Runtime;
use amethyst_rhusics::rhusics_core::WorldParameters;

#[derive(Default)]
pub struct InitState;

impl<'a, 'b> State<GameData<'a, 'b>, AllEvents> for InitState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.register::<ObjectType>();
        data.world.register::<Removal<RemovalId>>();
        data.world.add_resource(get_all_maps(&get_working_dir()));
        data.world.add_resource(AmbientColor(Rgba::from([0.1; 3])));
        data.world.add_resource(FutureProcessor::default());
        let hide_cursor = HideCursor { hide: false };
        data.world.add_resource(hide_cursor);

        let player_settings_data = std::fs::read_to_string(format!("{}/assets/base/config/player.ron",get_working_dir())).expect("Failed to read player.ron settings file.");
        let player_settings: PlayerSettings = ron::de::from_str(&player_settings_data).expect("Failed to load player settings from file.");

        let mut world_param = WorldParameters::<Vector3<f32>, f32>::default();//::new(Vector3::new(0, player_settings.gravity, 0));
        world_param = world_param.with_damping(1.0);
        data.world.add_resource(world_param);

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