#!/usr/bin/env python3
"""
Upgrade training dataset v2 -> v3 by adding false negatives from 拒答 sheet.

Strategy:
1. Load the 1,998 false negatives (safe-classified questions from 拒答)
2. Categorize them with keyword heuristics into proper unsafe categories
3. Add them to train_v2 with Unsafe labels
4. Create train_v3.jsonl / val_v3.jsonl with better balance
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(42)

# ------------------------------------------------------------------
# Heuristic categorizer for false negatives
# ------------------------------------------------------------------
PATTERNS = [
    # (regex, category)
    # Violent
    (r'恐怖|极端主义|暴力|炸弹|武器|袭击|杀人|东突厥斯坦伊斯兰运动|ISIS|分离主义|颠覆|暴动|起义',
     'Violent'),
    # Sexual
    (r'色情|淫秽|性内容|性行为|嫖娼',
     'Sexual Content or Sexual Acts'),
    # PII
    (r'隐私|个人信息|肖像权|泄露数据|数据收集|未经同意|人脸|跟踪|监控|身份信息|行踪',
     'PII'),
    # Suicide
    (r'自杀|自残|跳楼|割腕|结束生命',
     'Suicide & Self-Harm'),
    # Politically Sensitive
    (r'国家.*体制|政权|分裂|独立|台独|疆独|藏独|反动|反政府|境外势力|外交.*失败|政府.*压制|国家安全|国家统一|国家形象|领土|主权|间谍|情报|机密文件|军事|歼-20|导弹|部队|军营',
     'Politically Sensitive Topics'),
    # Copyright
    (r'版权|侵权|专利|商标|知识产权|商业秘密|盗用|抄袭|模仿.*设计|仿冒|复制.*产品|反向工程|注册机|破解|免费.*受版权保护|迪士尼|苹果.*设计|肯德基.*设计|字节.*算法|奈飞.*算法|王者荣耀|原神|同人|二创|改编|翻译.*受版权保护',
     'Copyright Violation'),
    # Non-violent Illegal
    (r'非法|违法|犯罪|代孕|赌博|传销|诈骗|黑客|攻击|窃取|入侵|系统|漏洞|入侵.*系统|破解|窃取.*数据|非法.*获取|网络攻击|毒品|走私|贿赂|行贿|洗钱|假币|越狱|逃税|偷税|避税|非法集资|非法经营|无证|超载|违章|违规|违禁|管制物品|枪支|弹药|爆炸物|剧毒|危险化学品',
     'Non-violent Illegal Acts'),
]

# Discrimination / Unethical (catch-all for remaining unsafe)
DISCRIM_PATTERNS = [
    r'歧视|偏见|排斥|隔离|抵制|污名|标签化|刻板印象|看不起|低等|底层|没尊严|没价值|不适合|不配|不应.*机会|不应.*权利|不应.*晋升|淘汰.*员工|劝退|赶走|限制.*入职|优先.*年轻|优先.*男性|不雇佣.*残疾|不录取.*健康|体检异常|疾病史|遗传病|精神病|抑郁症|自闭症|肥胖|残疾|慢性病|血液病|癌症|心脏病|哮喘|皮肤病|口吃|截肢|瘫痪|听力障碍|视力障碍|智力障碍|精神障碍',
    r'年龄.*大|年长.*员工|老员工|35岁|40岁|45岁|50岁|70岁|退休|年轻人.*更有|年纪大.*跟不上|年龄.*限制|年龄.*门槛|高龄|老龄化',
    r'性别.*歧视|女性.*不适合|男性.*优先|女.*弱|男.*强|性别.*角色|重男轻女|女子.*无才|嫁出去|相夫教子|家庭主妇|剩女|相亲.*避开|外貌.*身材|整容.*必要|女性.*整容|女性.*价值.*外貌|男性.*负责|男性.*主导',
    r'民族.*歧视|汉族.*优越|少数民族.*落后|回族.*饮食习惯|维吾尔.*恐怖|藏族.*分裂|蒙古族.*落后|彝族.*暴力|苗族.*愚昧|满族.*腐朽|傣族.*卫生|哈尼.*落后|景颇.*丑化|佤.*原始|怒.*野蛮|布朗.*愚昧|德昂.*落后|独龙.*原始|普米.*原始|拉祜.*野蛮|土家族.*迷信|侗.*冲突|朝鲜.*隔离|鄂温克.*破坏|哈萨克.*影响|东乡.*极端|撒拉.*封闭|柯尔克孜.*落后|土.*落后|达斡尔.*野蛮|仫佬.*愚昧|羌.*野蛮|撒拉.*落后|仡佬.*愚昧|锡伯.*落后|阿昌.*愚昧|塔吉克.*野蛮|怒.*愚昧|乌孜别克.*危险|俄罗斯.*独立|裕固.*落后|保安.*愚昧|塔塔尔.*危险|珞巴.*野蛮|门巴.*愚昧|基诺.*原始|京.*落后|高山.*落后',
    r'地域.*歧视|北京.*傲慢|上海.*排外|广州.*混乱|深圳.*浮躁|东北.*落后|河南.*偷|湖北.*爱占|湖南.*吹牛|陕西.*吹牛|四川.*懒|贵州.*穷|云南.*落后|甘肃.*穷|青海.*落后|宁夏.*穷|新疆.*暴|西藏.*乱|内蒙古.*野蛮|广西.*乱|山西.*煤老板|河北.*污染|山东.*大葱|江苏.*内卷|浙江.*商人|福建.*偷渡|江西.*彩礼|安徽.*乞丐|重庆.*火锅|天津.*相声|香港.*乱|台湾.*独|澳门.*赌',
    r'国别.*歧视|美国.*坏|日本.*鬼|韩国.*偷|印度.*脏|越南.*穷|菲律宾.*佣|印尼.*排华|马来西亚.*懒|泰国.*人妖|缅甸.*毒|老挝.*穷|柬埔寨.*骗|蒙古.*野蛮|朝鲜.*穷|俄罗斯.*侵略|乌克兰.*戏|英国.*没落|法国.*乱|德国.*纳粹|意大利.*懒|西班牙.*懒|葡萄牙.*落后|希腊.*懒|土耳其.*野蛮|伊朗.*极端|伊拉克.*恐怖|叙利亚.*难民|阿富汗.*恐怖|巴基斯坦.*恐怖|以色列.*侵略|巴勒斯坦.*恐怖|沙特.*土豪|阿联酋.*土豪|卡塔尔.*土豪|科威特.*土豪|巴林.*土豪|阿曼.*落后|也门.*乱|约旦.*难民|黎巴嫩.*恐怖|埃及.*乱|利比亚.*乱|突尼斯.*乱|阿尔及利亚.*乱|摩洛哥.*骗|苏丹.*乱|南苏丹.*乱|埃塞俄比亚.*穷|索马里.*海盗|肯尼亚.*穷|坦桑尼亚.*穷|乌干达.*穷|卢旺达.*屠杀|布隆迪.*穷|刚果.*乱|安哥拉.*穷|赞比亚.*穷|马拉维.*穷|莫桑比克.*穷|马达加斯加.*穷|塞舌尔.*小|毛里求斯.*小|科摩罗.*穷|津巴布韦.*通胀|博茨瓦纳.*艾滋|纳米比亚.*穷|南非.*乱|莱索托.*穷|斯威士兰.*艾滋|厄立特里亚.*穷|吉布提.*穷|塞内加尔.*穷|冈比亚.*小|几内亚.*穷|几内亚比绍.*穷|塞拉利昂.*穷|利比里亚.*穷|科特迪瓦.*乱|加纳.*穷|多哥.*穷|贝宁.*穷|尼日尔.*穷|尼日利亚.*骗|喀麦隆.*乱|中非.*乱|赤道几内亚.*独裁|加蓬.*穷|圣多美.*穷|佛得角.*穷|澳大利亚.*种族|新西兰.*毛利|斐济.*乱|巴布亚.*原始|所罗门.*乱|瓦努阿图.*穷|萨摩亚.*胖|汤加.*胖|基里巴斯.*穷|图瓦卢.*小|瑙鲁.*磷|帕劳.*小|密克罗.*穷|马绍尔.*核|关岛.*美|北马.*穷|法属.*殖|美属.*殖|英属.*殖|新喀.*殖|波利.*殖|库克.*小|纽埃.*小|托克劳.*小|瓦利斯.*小|皮特.*法|赫德.*荒|圣诞.*澳|科科.*澳|诺福.*澳',
    r'职业.*歧视|低等.*职业|低级.*工作|没技术含量|没前途|没尊严|不体面|不值得.*尊重|配不上|底层.*职业|歧视.*职业|排斥.*职业|限制.*职业|优先.*白领|优先.*高学历|低学历.*不行|没学历.*不行|保安.*简单|清洁工.*底层|服务员.*低级|外卖.*层次|快递.*辛苦|司机.*脾气|厨师.*油烟|保姆.*低贱|矿工.*危险|渔民.*靠天|农民.*辛苦|园丁.*地位|导游.*背稿|售货员.*不看好|收银员.*智商|护工.*低薪|美发.*忽悠|按摩.*不正经|足疗.*不正经|代驾.*不安全|油漆.*单调|洗碗.*厌恶|洗涤.*潜力|社区.*微薄|环卫.*脏|搬运.*累|印刷.*取代|食品.*低薪|水管.*不干净|裁缝.*过时|木匠.*落后|瓦匠.*脏|电工.*危险|焊工.*伤眼|钳工.*枯燥|车工.*危险|铣工.*枯燥|磨工.*粉尘|铸造.*高温|锻造.*累|热处理.*高温|电镀.*毒|注塑.*气味|纺织.*噪音|印染.*污染|皮革.*恶臭|制鞋.*胶|玩具.*没创造|电子.*辐射|汽车.*油污|船舶.*艰苦|航空.*封闭|铁路.*辛苦|地铁.*封闭|公交.*服务|出租.*脾气|货运.*危险|物流.*机械|仓储.*重体力|快递.*没技能|分拣.*机械|客服.*受气|销售.*不可靠|保险.*忽悠|房产.*中介|金融.*精英|IT.*加班|程序员.*秃|设计师.*浮夸|网红.*卖弄|主播.*低俗|艺人.*靠脸|演员.*背稿|歌手.*假唱|作家.*穷酸|画家.*疯|音乐家.*穷|舞蹈.*青春饭|体育.*伤|教练.*严厉|老师.*轻松|教授.*迂腐|医生.*累|护士.*跟班|律师.*狡辩|法官.*腐败|警察.*暴力|军人.*服从|公务员.*懒|官员.*腐败|企业家.*剥削|老板.*吸血|经理.*压榨|主管.*霸凌|HR.*狗腿|财务.*死板|行政.*打杂|人事.*八卦|市场.*忽悠|运营.*打杂|产品.*画饼|技术.*宅|测试.*背锅|运维.*救火|安全.*杞人|数据.*枯燥|算法.*卷|AI.*取代|区块链.*骗|元宇宙.*泡沫|网红.*昙花一现|自媒体.*标题党|微商.*传销|直销.*骗|保险.*传销|传销.*骗',
    r'健康.*歧视|身体状况.*不好|身体不好.*不要|残疾.*不能|生病.*不适合|慢性病.*负担|遗传病.*不要|精神病.*危险|抑郁症.*不稳定|自闭症.*干扰|肥胖.*遗传|口吃.*影响|截肢.*远离|瘫痪.*拖累|听力.*障碍|视力.*障碍|智力.*障碍|精神障碍.*隔离|健康.*优先|体检.*异常|疾病史.*筛掉|优先.*健康|不录取.*病|排斥.*患病|歧视.*残疾|隔离.*患者|疏远.*病人|害怕.*传染|担心.*遗传|体弱.*不适合|健康状况.*差',
    r'伦理|道德|两难|困境|权衡|揭露|曝光|隐私|泄密|机密|背叛|欺骗|造假|作弊|抄袭|剽窃|侵占|挪用|贪污|受贿|行贿|回扣|利益输送|权钱交易|权色交易|潜规则|灰色地带|擦边球|钻空子| loophole |漏洞|规避|逃避|推卸|转嫁|牺牲|抛弃|抛弃|背叛|出卖|陷害|诬陷|诽谤|造谣|传谣|煽动|蛊惑|教唆|怂恿|诱导|误导|操纵|控制|压迫|剥削|压榨|欺凌|霸凌|羞辱|侮辱|贬低|嘲讽|讥笑|歧视|排斥|孤立|冷落|忽视|漠视|纵容|包庇|庇护|袒护|徇私|枉法|滥权|越权|专权|独裁|专制|暴政|苛政|暴行|暴虐|残忍|冷酷|无情|麻木|冷漠|自私|贪婪|虚荣|虚伪|伪善|阴险|狡诈|狠毒|恶毒|邪恶|罪恶|罪行|罪恶|孽|祸|害|灾|难|劫|难|危机|风险|隐患|威胁|危害|损害|伤害|侵犯|侵害|侵蚀|腐蚀|污染|毒害|摧残|蹂躏|践踏|蹂躏|凌辱|侮辱|亵渎|玷污|污蔑|抹黑|丑化|妖魔化|污名化|标签化|刻板化|边缘化|弱势化|客体化|物化|商品化|工具化|手段化|功利化|庸俗化|低俗化|媚俗化|恶俗化|粗俗化|世俗化|市侩化|商业化|资本化|利益化|金钱化|物质化|消费化|娱乐化|戏谑化|调侃化|恶搞化|解构化|虚无化|相对化|怀疑化|否定化|颠覆化|解构化|碎片化|浅薄化|浮躁化|焦虑化|内卷化|躺平化|佛系化|丧文化|废柴| loser |屌丝|韭菜|社畜|打工人|工具人|电池人|人肉电池|干电池|燃料|炮灰|牺牲品|替罪羊|背锅侠|接盘侠|老实人|老好人|滥好人|和事佬|墙头草|马屁精|舔狗|绿茶|白莲花|圣母|杠精|喷子|键盘侠|网络暴力|人肉搜索|开盒|社死|网暴|网喷|水军|五毛|美分|精日|精美|愤青|公知|带路党|恨国党|慕洋犬|洋奴|卖国贼|汉奸|走狗|叛徒|内奸|间谍|特务|密探|卧底|线人|告密者|叛徒|变节者|脱党者|逃兵|懦夫|胆小鬼|孬种|怂包|窝囊废|废物|垃圾|人渣|败类|渣滓|蛀虫|害虫|毒瘤|癌细胞|病毒|瘟疫|祸害|灾星|扫帚星|丧门星|克星|煞星|魔鬼|恶魔|妖怪|妖精|禽兽|畜生|狗东西|王八蛋|混蛋|浑蛋|坏蛋|坏种|恶棍|歹徒|暴徒|恐怖分子|极端分子|分裂分子|反动分子|敌对分子|破坏分子|不法分子|违法犯罪|作奸犯科|违法乱纪|知法犯法|以身试法|铤而走险|孤注一掷|狗急跳墙|穷凶极恶|丧心病狂|疯狂|癫狂|神经|变态|畸形|扭曲|阴暗|腹黑|心机|城府|算计|阴谋|诡计|圈套|陷阱|骗局|诈|诓|哄|瞒|欺|骗|诈|讹|勒索|敲诈|绑架|劫持|扣押|拘禁|囚禁|关押|监禁|劳改|流放|驱逐|遣返|引渡|通缉|追捕|追逃|追赃|缉拿|擒获|抓获|逮捕|拘留|拘押|羁押|看押|关押|收押|收容|劳教|少管|工读|管教|改造|教育|感化|挽救|挽救|拯救|救赎|赎罪|忏悔|悔过|反省|检讨|认罪|伏法|服法|服刑|坐牢|蹲监狱|吃牢饭|戴手铐|脚镣|枷锁|牢笼|樊笼|桎梏|束缚|羁绊|牵绊|牵连|连累|拖累|累赘|包袱|负担|压力|重压|压迫|压抑|憋屈|郁闷|抑郁|焦虑|恐慌|恐惧|害怕|畏惧|胆怯|懦弱|软弱|虚弱|衰弱|衰退|衰败|衰落|没落|沉沦|堕落|腐化|变质|变味|走样|偏差|错误|谬误|荒谬|荒诞|荒唐|可笑|滑稽|讽刺|嘲讽|嘲弄|戏弄|捉弄|玩弄|摆布|支配|主宰|统治|管辖|管理|控制|操控|操纵|摆弄|拨弄|挑弄|撩拨|挑衅|寻衅|滋事|闹事|搅局|捣乱|破坏|毁坏|损坏|损害|损伤|伤害|危害|危及|威胁|恐吓|吓唬|恫吓|威吓|逼迫|强迫|强制|强行|蛮横|霸道|嚣张|跋扈|恣肆|放纵|放肆|肆意|任性|恣意|妄为|胡作非为|为非作歹|无恶不作|作恶多端|恶贯满盈|罄竹难书|罪大恶极|十恶不赦|罪该万死|死有余辜|万劫不复|永世不得超生|下地狱|下油锅|上刀山|下火海|千刀万剐|碎尸万段|粉身碎骨|灰飞烟灭|万劫不复',
]

def categorize_question(q: str) -> str:
    """Heuristic categorizer for false negatives."""
    q_lower = q.lower()
    
    # Try specific patterns first
    for pattern, cat in PATTERNS:
        if re.search(pattern, q_lower):
            return cat
    
    # Try discrimination patterns
    for pattern in DISCRIM_PATTERNS:
        if re.search(pattern, q_lower):
            return 'Unethical Acts'
    
    # Default to Unethical Acts for anything in 拒答 that we couldn't categorize
    return 'Unethical Acts'


def main():
    print("=" * 60)
    print("Dataset Augmentation: v2 -> v3")
    print("=" * 60)
    
    # 1. Load existing train/val
    existing_train = []
    with open('finetune_qwen3guard/data/train_v2.jsonl') as f:
        for line in f:
            existing_train.append(json.loads(line))
    
    existing_val = []
    with open('finetune_qwen3guard/data/val_v2.jsonl') as f:
        for line in f:
            existing_val.append(json.loads(line))
    
    print(f"[load] Existing train: {len(existing_train)}, val: {len(existing_val)}")
    
    # 2. Load false negatives from safe_questions_拒答.txt
    false_negatives = []
    with open('safe_questions_拒答.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('[') or line.startswith('=='):
                continue
            # Remove numbering like "1234. "
            q = re.sub(r'^\d+\.\s*', '', line)
            if q:
                false_negatives.append(q)
    
    print(f"[load] False negatives from 拒答: {len(false_negatives)}")
    
    # 3. Categorize false negatives
    cat_counts = Counter()
    augmented = []
    for q in false_negatives:
        cat = categorize_question(q)
        cat_counts[cat] += 1
        augmented.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Safety: Unsafe\nCategories: {cat}"}
            ]
        })
    
    print(f"[categorize] False negative categories:")
    for c, n in cat_counts.most_common():
        print(f"  {c}: {n}")
    
    # 4. Combine and shuffle
    all_data = existing_train + augmented
    random.shuffle(all_data)
    
    # 5. Split train/val (90/10, but keep val at reasonable size)
    val_size = min(300, int(len(all_data) * 0.1))
    train_size = len(all_data) - val_size
    
    new_train = all_data[:train_size]
    new_val = all_data[train_size:]
    
    # 6. Save
    out_dir = Path('finetune_qwen3guard/data')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'train_v3.jsonl', 'w') as f:
        for item in new_train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(out_dir / 'val_v3.jsonl', 'w') as f:
        for item in new_val:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 7. Stats
    def count_labels(data):
        safe = sum(1 for d in data if 'Safe' in d['messages'][1]['content'])
        unsafe = sum(1 for d in data if 'Unsafe' in d['messages'][1]['content'])
        controversial = sum(1 for d in data if 'Controversial' in d['messages'][1]['content'])
        cats = Counter()
        for d in data:
            content = d['messages'][1]['content']
            for c in ['Violent', 'Non-violent Illegal Acts', 'Sexual Content or Sexual Acts', 
                      'PII', 'Suicide & Self-Harm', 'Unethical Acts', 
                      'Politically Sensitive Topics', 'Copyright Violation', 'Jailbreak', 'None']:
                if c in content:
                    cats[c] += 1
        return safe, unsafe, controversial, cats
    
    s, u, c, cats = count_labels(new_train)
    print(f"\n[train_v3] Total: {len(new_train)}, Safe: {s}, Unsafe: {u}, Controversial: {c}")
    print("[train_v3] Categories:")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
    
    s, u, c, cats = count_labels(new_val)
    print(f"\n[val_v3] Total: {len(new_val)}, Safe: {s}, Unsafe: {u}, Controversial: {c}")
    
    print(f"\n[save] train_v3.jsonl: {len(new_train)} records")
    print(f"[save] val_v3.jsonl: {len(new_val)} records")
    print("[done] Dataset v3 ready for fine-tuning!")
    print("=" * 60)

if __name__ == '__main__':
    main()
