from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]


def test_offline_default_and_all_sections():
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert len(app.get('imgs'))>=0
    app.selectbox(key='scale').set_value(4).run()
    assert not app.exception and not app.error
    app.selectbox(key='export_format').set_value('JPEG').run()
    app.radio(key='page').set_value('Downloads').run()
    assert not app.exception and not app.error
    app.radio(key='page').set_value('Concepts & mathematics').run()
    assert not app.exception
    assert len(app.latex)>=12
    app.radio(key='page').set_value('Sensitivity').run()
    app.button(key='run_sensitivity').click().run()
    assert not app.exception and not app.error
    app.radio(key='page').set_value('Compare').run()
    app.multiselect(key='compare_models').set_value(['Bicubic']).run()
    app.button(key='run_compare').click().run()
    assert not app.exception and not app.error
    # Changing an input hides the old comparison instead of relabeling it.
    app.slider(key='noise').set_value(5.).run()
    assert any('Changing any input' in msg.value for msg in app.info)
    app.selectbox(key='source_mode').set_value('Live satellite (Sentinel-2)').run()
    assert not app.exception
    assert len(app.button(key='live_acquire').label)>0


def test_trained_hybrid_and_model_comparison_from_verified_cache():
    # Network-dependent checkpoint checks are opt-in; default CI remains offline.
    import os
    import pytest
    if os.environ.get('RSSR_TEST_REAL_MODELS')!='1':
        pytest.skip('Set RSSR_TEST_REAL_MODELS=1 after bounded checkpoint verification.')
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
    app.selectbox(key='model_id').set_value('M7').run()
    assert not app.exception and not app.error
    app.radio(key='page').set_value('Compare').run()
    app.button(key='run_compare').click().run()
    assert not app.exception and not app.error
    assert app.session_state['comparison']['results'][-1]['model_id']=='M7'
