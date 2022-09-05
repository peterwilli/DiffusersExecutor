import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        epilog=f'Use stable-diffusion',
        prog='python -m stable_diffusion_executor',
    )
    sps = parser.add_subparsers(dest='action', required=True)
    serve_parser = sps.add_parser('serve', help='Host stable-diffusion as service')
    serve_test_parser = sps.add_parser('serve_test', help='See if server is running as intended')
    for current_parser in [serve_parser, serve_test_parser]:
        current_parser.add_argument(
            'config_file',
            metavar='YAML_CONFIG_FILE',
            nargs='?',
            type=argparse.FileType('r'),
            help='The YAML config file to use, default is stdin.',
            default=sys.stdin,
        )
    upscale_parser = sps.add_parser('txt2img', help='Generate image based on prompt')
    upscale_parser.add_argument(
        'prompt',
        type=str,
        help='Prompt to use, default is stdin',
        default=sys.stdin,
    )
    upscale_parser.add_argument(
        'output_file',
        type=argparse.FileType('wb'),
        help='Output file, default is stdout',
        default=sys.stdout,
    )
    args = parser.parse_args()
    run_cli(args)

def serve_test(cfg):
    from jina import Document
    import time
    with _serve(cfg) as f:
        input = Document(text="A man in a dragon onesie jumping on his yacht")
        generated_da = f.post(on = '/stable_diffusion/txt2img', inputs = input)
        generated_da[0].load_uri_to_image_tensor()
        generated_da[0].save_image_tensor_to_file('serve_test_result.png')

def serve(cfg):
    with _serve(cfg) as f:
        f.block()

def _serve(cfg):
    from .txt2img_executor import Txt2ImgExecutor
    from jina import Flow
    return Flow.load_config(cfg)

def run_cli(args):
    if args.action == "serve":
        serve(args.config_file)
    elif args.action == "serve_test":
        serve_test(args.config_file)
    elif args.action == "txt2img":
        txt2img(args.prompt, args.output_file)

if __name__ == '__main__':
    main()