import glob
import os

files = glob.glob('scripts/*.py') + glob.glob('pipelines/*.py')
inject = "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"

for f in files:
    with open(f, 'r') as fp:
        content = fp.read()
    if 'sys.path.append' not in content:
        if content.startswith('"""'):
            end = content.find('"""', 3) + 3
            new_content = content[:end] + '\n' + inject + content[end:].lstrip()
        else:
            new_content = inject + content
        with open(f, 'w') as fp:
            fp.write(new_content)
