ROOT_DIR=$(dirname ${0})
cd $ROOT_DIR/..

pylint instant_ngp_3dml -j 4 --rcfile .vscode/.pylintrc

mypy instant_ngp_3dml --config-file .vscode/.mypy.ini

UNIMPORT_EXCLUDE="__init__.py"

if [ "$#" -eq 1 ]; then
    if [ "${1}" = "--force" ]; then
        unimport $(find instant_ngp_3dml -name "*.py") --exclude $UNIMPORT_EXCLUDE --remove && \
        reorder-python-imports $(find instant_ngp_3dml -name "*.py") --separate-relative
    fi
else
    echo "Display diff only, use --force to apply"
    unimport $(find instant_ngp_3dml -name "*.py") --exclude $UNIMPORT_EXCLUDE && \
    reorder-python-imports $(find instant_ngp_3dml -name "*.py") --separate-relative --diff-only
fi
