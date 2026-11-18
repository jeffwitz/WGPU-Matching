"""Reproduce the hole-moment experiment without overwriting archived results."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from hudge_strains_data import silhouette_proposal
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/home/jeff/CNRS/DIC/HudgeStrains/FEA'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('results/hudge_boundary_frontier_20261115/partial12/seeds.npz'))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    commands = []

    def run(script, *arguments):
        command = [sys.executable, str(scripts/script), *map(str, arguments)]
        commands.append(command)
        (root/'commands.json').write_text(json.dumps(commands, indent=2))
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)

    def solve(folder, data, baseline, proposals=None, validation=None):
        options = ['--output', folder, '--data', data, '--size', 12, '--stride', 4,
                   '--seed-archive', baseline, '--resume-exact']
        if proposals:
            options += ['--extra-seeds', proposals]
        if validation:
            options += ['--sector-validation', validation]
        run('qualify_hudge_boundary_frontier.py', *options)

    sources = ['hudge_hole_moments.py', 'benchmark_hudge_hole_moments.py',
               'benchmark_hudge_sector_curvature.py', 'qualify_hudge_boundary_frontier.py',
               'render_hudge_hole_twin.py', 'report_hudge_hole_moments.py',
               'run_hudge_hole_experiment.py', 'hudge_strains_data.py']
    provenance = {name: hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in sources}
    (root/'source_sha256.json').write_text(json.dumps(provenance, indent=2))
    solve(root/'control', args.data, args.baseline)
    for name, heldout in [('proposals', False), ('validated_proposals', True)]:
        flags = ['--heldout-sectors'] if heldout else []
        run('benchmark_hudge_hole_moments.py', '--output', root/name, '--data', args.data,
            '--baseline', args.baseline, '--sensitivity', *flags)
        solve(root/('validated' if heldout else 'moments'), args.data, args.baseline,
              root/name/'proposals.npz', root/name/'validation_supports.npz' if heldout else None)
    run('benchmark_hudge_sector_curvature.py', '--input', root/'validated_proposals',
        '--output', root/'quadratic_proposals')
    solve(root/'quadratic', args.data, args.baseline, root/'quadratic_proposals/proposals.npz',
          root/'quadratic_proposals/validation_supports.npz')
    for name, nonlinear in [('twin_affine', False), ('twin_nonlinear', True)]:
        folder = root/name
        run('render_hudge_hole_twin.py', '--output', folder, '--reference', args.data/'Img_F.tif',
            '--grid', args.baseline, *(['--nonlinear'] if nonlinear else []))
        variants = [('validated_proposals', True)] if nonlinear else [('proposals', False), ('validated_proposals', True)]
        for proposal_name, heldout in variants:
            run('benchmark_hudge_hole_moments.py', '--output', folder/proposal_name, '--data', folder,
                '--baseline', folder/'empty.npz', *(['--heldout-sectors'] if heldout else []))
            solve(folder/('validated' if heldout else 'moments'), folder, folder/'empty.npz',
                  folder/proposal_name/'proposals.npz',
                  folder/proposal_name/'validation_supports.npz' if heldout else None)
        grid = np.load(folder/'empty.npz')
        ids = np.flatnonzero(grid['selected'])
        images = [np.asarray(Image.open(folder/file)) for file in ['Img_F.tif', 'Img_wa_g191.tif']]
        silhouette, _ = silhouette_proposal(*images, grid['centres'][ids])
        truth = np.load(folder/'truth_scoring_only.npz')['parameters'][ids]
        for variant, parameters in [('silhouette', silhouette), ('oracle', truth)]:
            archive = folder/f'{variant}.npz'
            np.savez_compressed(archive, centres=grid['centres'][ids], parameters=parameters,
                                rank=np.zeros(len(ids), int))
            solve(folder/variant, folder, folder/'empty.npz', archive)
    run('report_hudge_hole_moments.py', '--root', root)


if __name__ == '__main__':
    main()
