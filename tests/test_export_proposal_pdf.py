from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'export_proposal_pdf.py'
spec = spec_from_file_location('export_proposal_pdf', MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_build_html_document_renders_cover_and_markdown_tables() -> None:
    html = module.build_html_document(
        '# BAB 1\n\n| Col | Val |\n| --- | --- |\n| A | B |\n\n![Chart](figures/chart.png)',
        title='Demo Title',
        subtitle='Demo Subtitle',
        source_name='proposal-final.md',
    )

    assert 'Demo Title' in html
    assert 'Demo Subtitle' in html
    assert 'proposal-final.md' in html
    assert '<table>' in html
    assert 'figures/chart.png' in html


def test_extract_heading_returns_first_markdown_h1() -> None:
    assert module.extract_heading('# Intro\n\n## Detail') == 'Intro'
    assert module.extract_heading('No heading here') is None
