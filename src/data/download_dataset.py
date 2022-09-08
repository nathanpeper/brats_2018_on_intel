# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import gdown

@click.command()
@click.argument('project_dir', type=click.Path(exists=True))
@click.argument('filename', type=click.Path())
@click.option('--url', default="https://drive.google.com/file/d/1BSmGx4IDEqebiU2WhKsy2Td6GjTlZfPV/view?usp=sharing")
@click.option('--output_path', default="data/raw", type=click.Path())
@click.option('--quiet', default=False)
@click.option('--fuzzy', default=True)

def main(project_dir, filename, url, output_path, quiet, fuzzy):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    project_dir = Path(project_dir)
    output_path = Path(output_path)
    output_str = str(Path(project_dir / output_path / filename))
    gdown.download(url=url, output=output_str, quiet=False, fuzzy=True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
