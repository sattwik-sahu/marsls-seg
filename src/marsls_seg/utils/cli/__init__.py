import typer
from marsls_seg.utils.cli.train import app as train_app


app = typer.Typer()
app.add_typer(train_app)
