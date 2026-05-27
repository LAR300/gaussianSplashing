# Gaussian Splashing — Setup Guide

Guia completo para reproduzir o paper [Gaussian Splashing](https://github.com/BGU-CS-VIL/gaussianSplashing) em ambiente local com GPU NVIDIA.

**Ambiente testado:**
- OS: Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
- Driver NVIDIA: 580.159.04 (suporta até CUDA 13.0)
- CUDA Toolkit instalado: 12.1
- Python: 3.8 (via Conda)
- PyTorch: 2.1.0+cu121

---

## Pré-requisitos

- Conda (Miniconda ou Anaconda) instalado
- Driver NVIDIA atualizado (`nvidia-smi` deve funcionar)
- ~12 GB de espaço em disco livre

---

## Passo 1 — Instalar o CUDA Toolkit 12.1 no sistema

> O Conda **não fornece os headers de compilação** necessários para compilar extensões CUDA customizadas. O toolkit precisa estar instalado no sistema via apt.

```bash
# Adicionar repositório oficial da NVIDIA (Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Instalar CUDA Toolkit 12.1
sudo apt-get install -y cuda-toolkit-12-1

# Confirmar instalação
/usr/local/cuda-12.1/bin/nvcc --version
```

> Para Ubuntu 20.04, troque `ubuntu2204` por `ubuntu2004` na URL do wget.

---

## Passo 2 — Instalar o COLMAP

```bash
sudo apt-get install -y colmap
```

---

## Passo 3 — Clonar o repositório com submodulos

```bash
cd ~/Documents

# O --recursive é obrigatório para clonar os submodulos CUDA
git clone https://github.com/BGU-CS-VIL/gaussianSplashing.git --recursive
cd gaussianSplashing

# Confirmar que os submodulos foram clonados
ls submodules/diff-gaussian-rasterization_UW/
ls submodules/simple-knn/
```

---

## Passo 4 — Criar o ambiente Conda

> Não usar o `environment.yml` do repositório — ele causa conflitos de dependência. Criar o ambiente manualmente.

```bash
conda create -n gaussianSplashing_env python=3.8 -y
conda activate gaussianSplashing_env

# Instalar PyTorch com CUDA 12.1 via pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verificar que a GPU é detectada
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Esperado: True + nome da GPU
```

---

## Passo 5 — Instalar dependências Python

```bash
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install opencv-python==4.8.1.78
pip install matplotlib
pip install Pillow
pip install tqdm
pip install plyfile
pip install scikit-learn
pip install wandb
```

> As versões de `numpy` e `scipy` importam — `scipy 1.10.x + numpy 1.24.x` é a combinação estável. Versões mais recentes causam `ValueError` na importação do `scipy.special`.

---

## Passo 6 — Compilar os submodulos CUDA

```bash
# Configurar CUDA_HOME para apontar para o toolkit do sistema
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Confirmar que está usando o nvcc correto
which nvcc        # deve retornar /usr/local/cuda-12.1/bin/nvcc
nvcc --version    # deve dizer "release 12.1"

# Compilar o rasterizador underwater (submodulo principal)
cd ~/Documents/gaussianSplashing/submodules/diff-gaussian-rasterization_UW
pip install -e .

# Compilar o simple-knn
cd ../simple-knn
pip install -e .

cd ~/Documents/gaussianSplashing
```

---

## Passo 7 — Tornar o CUDA_HOME permanente no ambiente

```bash
conda env config vars set CUDA_HOME=/usr/local/cuda-12.1
conda env config vars set PATH=/usr/local/cuda-12.1/bin:$PATH
conda env config vars set LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Reativar para aplicar
conda deactivate
conda activate gaussianSplashing_env

echo $CUDA_HOME   # deve retornar /usr/local/cuda-12.1
```

---

## Passo 8 — Testar os módulos compilados

```bash
python -c "
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
print('Todos os modulos CUDA carregados com sucesso!')
"
```

---

## Passo 9 — Corrigir bug no camera_utils.py

O repositório tem um bug de sintaxe na linha 23 de `utils/camera_utils.py`: um `.` no lugar de uma vírgula ao calcular a resolução das câmeras. Corrija antes de rodar o treino:

```bash
sed -i 's/resolution_scale \* args\.resolution))\. round/resolution_scale * args.resolution)), round/' utils/camera_utils.py

# Confirmar a correção (deve mostrar uma vírgula separando os dois round())
sed -n '20,26p' utils/camera_utils.py
```

---

## Passo 10 — Ajustar o config para GPUs com 6 GB de VRAM

O modo de densificação padrão (`OFF`) não tem limite de gaussians e estoura a VRAM em GPUs com menos de 8 GB. Troque para o modo MCMC, que respeita um teto máximo:

```bash
sed -i 's/densification: "OFF"/densification: "MCMC"/' configs/default.yaml
sed -i 's/cap_max: 300000/cap_max: 500000/' configs/default.yaml
```

O `cap_max: 500000` limita o modelo a 500k gaussians — seguro para 6 GB de VRAM. Aumente para `700000` se tiver 8 GB.

---

## Uso — Treinar com um dataset

### Preparar o dataset (COLMAP)

O dataset deve conter imagens e o arquivo `database.db` gerado pelo COLMAP. O script `convert.py` executa a reconstrução esparsa e gera a pasta `sparse/0/` necessária.

```bash
# A flag QT_QPA_PLATFORM=offscreen é necessária para rodar sem interface gráfica
QT_QPA_PLATFORM=offscreen python convert.py -s assets/south-building

# Confirmar estrutura gerada
ls assets/south-building/sparse/0/
# Esperado: cameras.bin  images.bin  points3D.bin
```

### Treinar

```bash
QT_QPA_PLATFORM=offscreen PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train.py -s assets/south-building -r 4
```

- `QT_QPA_PLATFORM=offscreen` — evita crash do Qt em ambientes sem display
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` — reduz fragmentação de VRAM
- `-r 4` — divide a resolução das imagens por 4, reduzindo uso de memória (use `-r 2` se tiver mais de 8 GB de VRAM)

A saída é salva automaticamente em `./output/` com um nome gerado a partir das configurações do experimento.

### Renderizar imagens estáticas

```bash
# Criar o symlink necessário (o render.py procura cfg_args sem extensão)
cd output/<nome_da_pasta_do_modelo>/
ln -s cfg_args.json cfg_args
cd ~/Documents/gaussianSplashing

# Renderizar
QT_QPA_PLATFORM=offscreen python render.py -m output/<nome_da_pasta_do_modelo> --skip_train
```

As imagens renderizadas ficam em `output/<nome_da_pasta_do_modelo>/test/ours_15000/renders/`.

### Visualizar com SIBR (interativo)

Instalar dependências de compilação:

```bash
sudo apt-get install -y cmake libglew-dev libglfw3-dev libglm-dev libopencv-dev \
    libassimp-dev libboost-all-dev libssl-dev libavcodec-dev libavformat-dev \
    libavutil-dev libswscale-dev libembree-dev
```

Compilar o SIBR:

```bash
cd ~/Documents/gaussianSplashing/SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install --parallel 4
```

Rodar o visualizador:

```bash
./SIBR_viewers/bin/SIBR_gaussianViewer_app -m output/<nome_da_pasta_do_modelo>
```

---

## Problemas Conhecidos

| Erro | Causa | Solução |
|---|---|---|
| `cuda_runtime.h: No such file or directory` | CUDA Toolkit não instalado no sistema | Executar Passo 1 |
| `cuda/std/type_traits: No such file or directory` | Headers CCCL ausentes | Garantir que o toolkit completo foi instalado via apt |
| `SyntaxError` no torch ao ativar Conda | Arquivo do PyTorch corrompido | Recriar o ambiente do zero |
| `images.bin` / `images.txt` not found | Pasta `sparse/0/` ausente no dataset | Executar `convert.py` antes do `train.py` |
| `Could not load Qt platform plugin "xcb"` | Qt tentando abrir janela gráfica | Usar `QT_QPA_PLATFORM=offscreen` em todos os comandos |
| `AttributeError: 'int' object has no attribute 'round'` | Bug de sintaxe no `camera_utils.py` | Executar Passo 9 |
| `CUDA out of memory` durante densificação | Modelo crescendo sem limite de gaussians | Executar Passo 10 (modo MCMC) |
| `CUDA out of memory` no carregamento | Pouca RAM disponível | Fechar Chrome/Slack/Spotify e usar `-r 4` |
| `cfg_args: No such file or directory` | `render.py` procura arquivo sem extensão | Criar symlink: `ln -s cfg_args.json cfg_args` |
| `wandb` pedindo login | Primeira execução do W&B | Criar conta em wandb.ai ou escolher opção 3 (sem visualização) |

---

## Preparando um Dataset Customizado

Esta seção cobre como transformar dados brutos em um dataset compatível com o repositório, partindo de quatro pontos de partida diferentes. Em todos os casos, o objetivo final é a mesma estrutura:

```
meu_dataset/
├── images/          ← imagens originais
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── database.db
```

> **Contexto subaquático:** ambientes de baixa visibilidade, turvo ou com partículas em suspensão reduzem drasticamente a quantidade de features detectadas pelo COLMAP. Parâmetros específicos para esse cenário estão marcados com 🌊.

---

### Fluxo A — Só tenho imagens (fotos ou frames)

**Pré-requisito:** pasta com imagens `.jpg` ou `.png` cobrindo a cena de múltiplos ângulos. Quanto mais sobreposição entre imagens adjacentes, melhor a reconstrução.

**Boas práticas de captura:**
- Mínimo de 30-50% de sobreposição entre imagens consecutivas
- Evitar movimentos bruscos entre frames
- 🌊 Em ambiente subaquático: preferir iluminação frontal, evitar contra-luz, capturar mais imagens (maior sobreposição compensa a baixa visibilidade)

**Passo 1 — Organizar as imagens:**
```bash
mkdir -p meu_dataset/images
cp suas_imagens/*.jpg meu_dataset/images/
```

**Passo 2 — Rodar o pipeline completo do COLMAP via convert.py:**
```bash
QT_QPA_PLATFORM=offscreen python convert.py -s meu_dataset
```

O `convert.py` executa automaticamente: extração de features → matching → reconstrução esparsa → undistortion.

**🌊 Se o COLMAP falhar ou gerar poucos pontos em ambiente subaquático**, rode manualmente com parâmetros ajustados:

```bash
# Extração de features com mais octaves (detecta mais features em imagens turvas)
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
    --database_path meu_dataset/database.db \
    --image_path meu_dataset/images \
    --ImageReader.single_camera 1 \
    --SiftExtraction.num_octaves 4 \
    --SiftExtraction.peak_threshold 0.003 \
    --SiftExtraction.edge_threshold 10

# Matching exaustivo (mais lento, mas mais robusto para cenas difíceis)
QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher \
    --database_path meu_dataset/database.db

# Reconstrução esparsa
mkdir -p meu_dataset/sparse
QT_QPA_PLATFORM=offscreen colmap mapper \
    --database_path meu_dataset/database.db \
    --image_path meu_dataset/images \
    --output_path meu_dataset/sparse

# Undistortion (necessário para o train.py)
QT_QPA_PLATFORM=offscreen colmap image_undistorter \
    --image_path meu_dataset/images \
    --input_path meu_dataset/sparse/0 \
    --output_path meu_dataset \
    --output_type COLMAP
```

---

### Fluxo B — Tenho um vídeo e preciso extrair os frames

**Pré-requisito:** arquivo de vídeo `.mp4`, `.mov` ou similar. O FFmpeg é usado para extrair os frames.

**Passo 1 — Instalar o FFmpeg:**
```bash
sudo apt-get install -y ffmpeg
```

**Passo 2 — Extrair frames do vídeo:**
```bash
mkdir -p meu_dataset/images

# Extrair 2 frames por segundo (ajuste conforme a velocidade do vídeo)
ffmpeg -i seu_video.mp4 -qscale:v 1 -qmin 1 -vf fps=2 meu_dataset/images/%04d.jpg
```

> Dica: para vídeos com câmera lenta ou cenas densas, use `fps=1`. Para vídeos com movimento rápido, use `fps=4` ou mais. O objetivo é ter sobreposição suficiente sem imagens redundantes demais (que aumentam o tempo do COLMAP sem ganho).

> 🌊 Em vídeos subaquáticos, prefira `fps=3` ou `fps=4` para compensar a baixa qualidade de features por frame.

**Passo 3 — Seguir o Fluxo A** a partir daqui, usando `meu_dataset/` como pasta de entrada.

---

### Fluxo C — Tenho imagens + dados de pose (GPS/IMU)

**Pré-requisito:** imagens + arquivo com poses conhecidas (GPS, IMU, ou poses exportadas de outro sistema como ROS2 ou SLAM). Esse fluxo pula a etapa de estimativa de pose do COLMAP, tornando a reconstrução mais rápida e robusta.

**Passo 1 — Converter as poses para o formato COLMAP:**

O COLMAP espera um arquivo `images.txt` com o seguinte formato por linha:

```
IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
```

Onde `QW QX QY QZ` é o quaternion de rotação e `TX TY TZ` é a translação. Exemplo:

```
1 0.9997 0.0131 -0.0208 0.0020 -0.3509 0.1624 1.5932 1 P1180143.JPG
2 0.9996 0.0148 -0.0213 0.0018 -0.2935 0.1611 1.5789 1 P1180145.JPG
```

**Passo 2 — Criar o arquivo cameras.txt:**

```
# Camera list with one line of data per camera:
# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
1 PINHOLE 1920 1080 1500.0 1500.0 960.0 540.0
```

Substitua `WIDTH`, `HEIGHT` e os parâmetros intrínsecos (`fx fy cx cy`) pelos valores reais da sua câmera.

**Passo 3 — Criar a estrutura e rodar triangulação:**
```bash
mkdir -p meu_dataset/sparse/0
cp cameras.txt meu_dataset/sparse/0/
cp images.txt meu_dataset/sparse/0/
# Criar points3D.txt vazio (será preenchido pela triangulação)
touch meu_dataset/sparse/0/points3D.txt

# Extrair features (ainda necessário para triangulação)
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
    --database_path meu_dataset/database.db \
    --image_path meu_dataset/images

# Matching
QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher \
    --database_path meu_dataset/database.db

# Importar poses conhecidas e triangular pontos 3D
QT_QPA_PLATFORM=offscreen colmap point_triangulator \
    --database_path meu_dataset/database.db \
    --image_path meu_dataset/images \
    --input_path meu_dataset/sparse/0 \
    --output_path meu_dataset/sparse/0

# Undistortion
QT_QPA_PLATFORM=offscreen colmap image_undistorter \
    --image_path meu_dataset/images \
    --input_path meu_dataset/sparse/0 \
    --output_path meu_dataset \
    --output_type COLMAP
```

---

### Fluxo D — Tenho um dataset COLMAP parcial (tem database.db mas sem sparse/)

Este foi o caso do dataset `south-building` usado no teste desta documentação. O `database.db` já contém as features extraídas, faltando apenas a reconstrução esparsa.

**Verificar o que está presente:**
```bash
ls meu_dataset/
# Esperado: database.db  images/
```

**Rodar apenas a reconstrução:**
```bash
QT_QPA_PLATFORM=offscreen python convert.py -s meu_dataset
```

O `convert.py` detecta automaticamente a presença do `database.db` e pula a extração de features, rodando apenas o mapper e o undistorter.

**Confirmar resultado:**
```bash
ls meu_dataset/sparse/0/
# Esperado: cameras.bin  images.bin  points3D.bin
```

---

### Verificar a qualidade da reconstrução

Independente do fluxo usado, vale checar a qualidade antes de treinar:

```bash
# Ver quantas imagens e pontos foram reconstruídos
QT_QPA_PLATFORM=offscreen colmap model_analyzer \
    --path meu_dataset/sparse/0
```

Valores de referência para uma boa reconstrução:
- **Imagens registradas:** >80% do total
- **Pontos 3D:** >10.000 para cenas pequenas, >50.000 para cenas grandes
- **Mean track length:** >3.0 (quantas imagens cada ponto 3D aparece em média)
- **Mean reproj. error:** <1.5 pixels

> 🌊 Em ambientes subaquáticos, valores menores são esperados devido à baixa visibilidade. Uma reconstrução com 60% das imagens registradas e erro de reprojeção de até 2.0 pixels ainda pode produzir resultados aceitáveis para o Gaussian Splashing.

---

### Treinar com o dataset customizado

Após qualquer um dos fluxos acima, o comando de treino é o mesmo:

```bash
QT_QPA_PLATFORM=offscreen PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train.py -s meu_dataset -r 4
```

---

## Troubleshooting — Memória e Performance

### Processo morto com "Killed" sem mensagem de erro

**Sintoma:** o terminal mostra `Killed` abruptamente durante o carregamento das câmeras ou início do treino, sem stack trace.

**Causa:** o OOM Killer do Linux encerrou o processo por falta de RAM disponível. Não é um bug do código.

**Diagnóstico:**
```bash
# Ver memória disponível
free -h

# Ver o que está consumindo RAM
ps aux --sort=-%mem | head -15
```

**Solução:**

1. Fechar aplicações pesadas antes de treinar — Chrome, Slack e Spotify juntos podem consumir 6-8 GB de RAM:
```bash
# Verificar processos Chrome especificamente
ps aux | grep chrome | wc -l
```

2. Rodar o treino com resolução reduzida:
```bash
# -r 2: divide resolução por 2 (recomendado para 16 GB RAM)
# -r 4: divide resolução por 4 (recomendado para 8-12 GB RAM)
QT_QPA_PLATFORM=offscreen PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train.py -s meu_dataset -r 4
```

3. Se o sistema estiver usando muito swap, um reboot antes de treinar ajuda a liberar memória fragmentada:
```bash
free -h
# Se "used" em Swap estiver acima de 4 GB, considere reiniciar
```

---

### CUDA out of memory durante o treino

**Sintoma:** erro `torch.cuda.OutOfMemoryError` em algum ponto do treino, geralmente durante `densify_and_split` ou `loss.backward()`.

**Causa:** o número de gaussians cresceu além da capacidade da VRAM. Por padrão, o densificador não tem limite — o modelo pode crescer de 86k para 2-3M gaussians ao longo do treino.

**Solução 1 — Usar modo MCMC com teto de gaussians (recomendado):**
```bash
# Editar configs/default.yaml
sed -i 's/densification: "OFF"/densification: "MCMC"/' configs/default.yaml
sed -i 's/cap_max: 300000/cap_max: 500000/' configs/default.yaml
```

Referência de `cap_max` por quantidade de VRAM:

| VRAM disponível | cap_max recomendado |
|---|---|
| 4 GB | 300.000 |
| 6 GB | 500.000 |
| 8 GB | 700.000 |
| 12 GB+ | 1.000.000+ |

**Solução 2 — Reduzir resolução das imagens:**
```bash
# Usar -r 4 no comando de treino
python train.py -s meu_dataset -r 4
```

**Solução 3 — Reduzir fragmentação de VRAM:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

As três soluções podem e devem ser combinadas em GPUs com 6 GB ou menos.

---

### Treino lento ou GPU subutilizada

**Diagnóstico:**
```bash
# Monitorar uso de GPU em tempo real (em outro terminal)
watch -n 1 nvidia-smi
```

Se `GPU-Util` estiver abaixo de 50% durante o treino, o gargalo é a CPU/RAM carregando as imagens, não a GPU processando. Isso acontece quando `-r 1` (resolução original) é usado com imagens de alta resolução.

**Solução:** usar `-r 2` ou `-r 4` para reduzir o tamanho das imagens carregadas na memória, o que paradoxalmente pode acelerar o treino por reduzir o gargalo de I/O.

---

### Referência rápida de recursos por configuração

| Configuração | RAM necessária | VRAM necessária | Tempo estimado (RTX 3050) |
|---|---|---|---|
| `-r 1` + 128 imagens | ~14 GB | >6 GB | Inviável em 6 GB |
| `-r 2` + 128 imagens | ~8 GB | ~5 GB | 20-30 min |
| `-r 4` + 128 imagens | ~5 GB | ~3 GB | 25-35 min |
| `-r 4` + 30 imagens | ~3 GB | ~2 GB | 10-15 min |

> Estes valores são estimativas baseadas no dataset `south-building` (128 imagens, cena externa). Datasets subaquáticos com imagens de menor resolução podem exigir menos recursos.