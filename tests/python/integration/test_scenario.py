from dku_plugin_test_utils import dss_scenario

TEST_PROJECT_KEY = "PLUGINTESTAIART"


def test_run_txt2img(user_dss_clients):
    dss_scenario.run(
        user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TXT2IMG"
    )


def test_run_img2img(user_dss_clients):
    dss_scenario.run(
        user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="IMG2IMG"
    )
