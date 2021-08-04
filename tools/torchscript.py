import argparse
import torch
import tsai.all as tmsr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of model class, should be able to be got by `getattr(tmsr, args.model)`')
    parser.add_argument('weight', type=str, help='Path to ')
    parser.add_argument('-t', type=str, default='script', choices=('script', 'trace'),
                        help='Tracing can handle anything that uses only PyTorch Tensors and operations '
                             'except for control flow (e.g. if/for), while scripting requires the features '
                             'in your model supported by the compiler. See more in "https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#basics-of-torchscript".')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    model = getattr(tmsr, args.model)(1, 17)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval()

    # # Tracing
    # module = Module()
    # x = torch.rand(3, 4)
    # traced_cell = torch.jit.trace(module, x)

    # # Scripting
    # module = Module()
    # scripted_cell = torch.jit.script(module)

    print('Torchscript begins...')
    if args.t == 'script':
        scripted = torch.jit.script(model)
    elif args.t == 'trace':
        scripted = torch.jit.trace(model, torch.rand(1, 1, 600))
    print('Torchscript ends.')

    scripted.save('save/scripted_' + args.weight.split('/')[-1])
    print('Scripted saved to save/scripted_' + args.weight.split('/')[-1])
    print('\n', scripted.code)
