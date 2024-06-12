Примерная структура:

1. Объявление композитной сети (как суперсети, так и подсети).
    Composite(
        backbone: Backbone | str,
        neck: Neck | str,
        head: Head | str
    )
    Скорее всего, для объявления композитной подсети нельзя в качестве аргументов частей подавать строки.
    

2. Наследование классов суперсети и подсети, а также пояснение, где динамичность.
    - Composite_super(Composite_sub)
        - состоит из динамических подсетей
    - Backbone_super(Backbone_sub)
        - OFA
    - Neck_super(Neck_sub)
        - conv1x1: динамический размер входных каналов для выходов из ResNet-backbone (обязательно, если выходы backbone динамические)
        - conv1x1: динамический размер выходных каналов (может быть)
    - Head_super(Head_sub)
        - динамические размеры входных каналов свёрток (обязательно, если выходы шеи динамические)
        - динамические свёрточные слои (может быть)

3. Варианты частей
    - Backbones
        - MobileNet
        - ResNet
    - Necks (опционально)
        - FPN
    - Heads
        - Classification
        - Segmentation
        - Keypoints detection
        - Object detection

4. Все классы
    - Composites
        - Composite_super, Composite_sub
    - Backbones
        - BaseBackbone_super, BaseBackbone_sub ? : -
        - MobileNet_super, MobileNet_sub
        - ResNet_super, ResNet_sub
    - Necks
        - FPN_super, FPN_sub
    - Heads
        - BaseHead_super, BaseHead_sub ? : +
        - Classification_super, Classification_sub
        - Segmentation_super, Segmentation_sub
        - Keypoints_super, Keypoints_sub
        - Detection_super, Detection_sub

5. Hooks, кажется, нужны только для backbone. Можно создавать класс HooksCatcher(Backbone) при инициализации композитной сети.







