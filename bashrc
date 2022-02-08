if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

if [ -f ~/.bash_aliases ]; then
	. ~/.bash_aliases
fi

if [ -f ~/.bash_export ]; then
        . ~/.bash_export
fi

if [ -f ~/.bash_conda ]; then
        . ~/.bash_conda
fi

if [ -f ~/.bash_modules ]; then
        . ~/.bash_modules
fi


# TMUX
#export CPATH=/home/18me92r07/Apps/local/include:$CPATH
#export PKG_CONFIG_PATH=/home/18me92r07/Apps/local/lib/pkgconfig:$PKG_CONFIG_PATH
#export LD_LIBRARY_PATH=/home/18me92r07/Apps/local/lib:$LD_LIBRARY_PATH
#export PATH=/home/18me92r07/Apps/local/bin:$PATH


cd /home/j20210241/AI_CFD/poisson_CNN/poisson_CNN
